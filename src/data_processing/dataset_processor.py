import os
from functools import partial

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from src.helper.logging import logger


class PretrainingDatasetProcessor:
    """Dataset processor for preparing datasets for pretraining with packing."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        text_column: str = "text",
        num_proc: int = 1,
        seed: int = 42,
        use_packing: bool = True,
    ):
        """Initialize the dataset processor.

        Args:
            tokenizer: Tokenizer to use for tokenization
            max_seq_length: Maximum sequence length for tokenization
            text_column: Name of the text column in the dataset
            num_proc: Number of processes for parallel processing
            seed: Random seed for reproducible splits and shuffling
            use_packing: Whether to use packing (concatenation + chunking) for efficiency

        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_column = text_column
        self.num_proc = num_proc
        self.seed = seed
        self.use_packing = use_packing

        # Validate tokenizer has required tokens
        if self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer must have a BOS token defined for pretraining.")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token defined for pretraining.")

    def prepare_dataset(
        self,
        path: str,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
    ) -> dict[str, Dataset]:
        """Load and prepare dataset for pretraining.

        Args:
            path: Path to local parquet file or HuggingFace dataset name
            n_train: Number of training records to use (None for all)
            n_val: Number of validation records to use (None for all)
            n_test: Number of test records to use (None for all)

        Returns:
            Dictionary containing tokenized and packed dataset splits

        """
        # Step 1: Load the dataset
        ds = self._load_raw_dataset(path=path)

        # Step 2: Create evaluation split if needed (before limiting records)
        if n_val or n_test:
            ds = self._create_train_eval_split(ds=ds, n_val=n_val, n_test=n_test)

        # Step 3: Limit records if specified (after splitting)
        if n_train is not None and n_train > 0:
            ds = self._limit_splits(ds=ds, n_train=n_train, n_val=n_val, n_test=n_test)

        # Step 4: Tokenize with EOS separator (without special tokens)
        tokenized_ds = self._tokenize_datasets(ds)

        # Step 5: Apply packing if enabled
        if self.use_packing:
            logger.info(f"Applying packing to group sequences into {self.max_seq_length} token blocks...")
            return self._pack_datasets(tokenized_ds)
        else:
            logger.warning("Packing disabled - using truncation (less efficient for pretraining)")
            return self._truncate_datasets(tokenized_ds)

    def _load_raw_dataset(self, path: str) -> dict[str, Dataset]:
        """Load dataset from local parquet or HuggingFace hub."""
        if self._is_local_parquet(path):
            ds = load_dataset("parquet", data_files=path, split="train")
            return {"train": ds}
        else:
            return load_dataset(path)

    @staticmethod
    def _is_local_parquet(path: str) -> bool:
        """Check if the path is a local parquet file."""
        return os.path.exists(path) and (path.endswith(".parquet") or path.endswith(".parquet."))

    def _create_train_eval_split(
        self,
        ds: dict[str, Dataset],
        n_val: int | None,
        n_test: int | None,
    ) -> dict[str, Dataset]:
        """Create train/eval split from the train dataset."""
        if "train" not in ds:
            logger.warning("No 'train' split found in dataset. Returning dataset as-is.")
            return ds

        # Shuffle the dataset before splitting
        ds["train"] = ds["train"].shuffle(seed=self.seed)

        train_size = len(ds["train"])
        total_split_size = (n_val or 0) + (n_test or 0)

        if total_split_size == 0:
            return ds

        if train_size <= total_split_size:
            logger.warning(
                f"Cannot create splits: dataset has {train_size} records but {total_split_size} needed for splits. "
                "Splits will be limited to available data."
            )

        # Create validation split if requested
        if n_val and n_val > 0:
            val_size = min(n_val, train_size)
            split_ds = ds["train"].train_test_split(
                test_size=val_size,
                shuffle=False,
                seed=self.seed,
            )
            ds["train"] = split_ds["train"]
            ds["validation"] = split_ds["test"]
            logger.info(f"Created validation split with {len(ds['validation'])} records")

        # Create test split if requested
        if n_test and n_test > 0:
            test_size = min(n_test, len(ds["train"]))
            split_ds = ds["train"].train_test_split(
                test_size=test_size,
                shuffle=False,
                seed=self.seed,
            )
            ds["train"] = split_ds["train"]
            ds["test"] = split_ds["test"]
            logger.info(f"Created test split with {len(ds['test'])} records")

        logger.info(
            f"Final splits - Train: {len(ds['train'])}, "
            + (f"Val: {len(ds['validation'])}, " if "validation" in ds else "")
            + (f"Test: {len(ds['test'])}" if "test" in ds else "")
        )

        return ds

    def _limit_splits(
        self,
        ds: dict[str, Dataset],
        n_train: int | None,
        n_val: int | None,
        n_test: int | None,
    ) -> dict[str, Dataset]:
        """Limit the number of records in each split."""
        limits = {
            "train": n_train,
            "validation": n_val,
            "test": n_test,
        }

        for split_name in list(ds.keys()):
            if split_name in limits and limits[split_name] is not None and limits[split_name] > 0:
                original_size = len(ds[split_name])
                limit = min(limits[split_name], original_size)
                ds[split_name] = ds[split_name].select(range(limit))
                logger.info(f"Limited {split_name} split from {original_size} to {limit} records")

        splits_info = ", ".join([f"{name}: {len(split)}" for name, split in ds.items()])
        logger.info(f"Final dataset sizes - {splits_info}")

        return ds

    def _tokenize_datasets(self, ds: dict[str, Dataset]) -> dict[str, Dataset]:
        """Tokenize all dataset splits and add EOS separator between documents."""
        tokenized_ds = {}
        for split_name, split_data in ds.items():
            tokenized_ds[split_name] = split_data.map(
                self._tokenize_batch,
                batched=True,
                num_proc=max(1, self.num_proc),
                remove_columns=split_data.column_names,
                desc=f"Tokenizing and adding EOS for {split_name}",
            )
        return tokenized_ds

    def _tokenize_batch(self, examples: dict) -> dict:
        """Tokenize a batch of examples and add EOS separator.

        This tokenization step:
        1. Disables automatic special token addition (add_special_tokens=False)
        2. Adds EOS token manually after each document as separator
        3. Does NOT truncate (truncation will happen during packing)
        """
        # Tokenize without automatic special tokens
        tokenized_output = self.tokenizer(
            examples[self.text_column],
            add_special_tokens=False,
            truncation=False,
        )

        # Add EOS as separator after each document
        eos_id = self.tokenizer.eos_token_id
        tokenized_output["input_ids"] = [ids + [eos_id] for ids in tokenized_output["input_ids"]]

        # Handle attention_mask if present
        if "attention_mask" in tokenized_output:
            tokenized_output["attention_mask"] = [mask + [1] for mask in tokenized_output["attention_mask"]]

        return tokenized_output

    def _pack_datasets(self, tokenized_ds: dict[str, Dataset]) -> dict[str, Dataset]:
        """Apply packing to all dataset splits."""
        packed_ds = {}
        for split_name, split_data in tokenized_ds.items():
            packed_ds[split_name] = split_data.map(
                partial(
                    self._group_texts,
                    max_seq_length=self.max_seq_length,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                ),
                batched=True,
                num_proc=max(1, self.num_proc),
                desc=f"Packing {split_name} into {self.max_seq_length}-token blocks",
            )
            logger.info(f"Packed {split_name}: {len(packed_ds[split_name])} blocks of {self.max_seq_length} tokens")
        return packed_ds

    @staticmethod
    def _group_texts(
        examples: dict[str, list],
        max_seq_length: int,
        bos_token_id: int,
        eos_token_id: int,
    ) -> dict[str, list]:
        """Concatenate all sequences into one long list and chunk into fixed-length blocks.

        Final structure of each block:
        [<bos>, doc1_content...<eos>, doc2_content...<eos>, ...]

        Args:
            examples: Batch of tokenized examples (each document ends with EOS)
            max_seq_length: Target sequence length for each block (including BOS)
            bos_token_id: BOS token ID to prepend at start of each block
            eos_token_id: EOS token ID (kept for compatibility)

        Returns:
            Dictionary with packed sequences

        """
        # 1. Concatenate all texts into one super-long list
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # 2. Calculate chunk size (reserve 1 token for BOS)
        chunk_size = max_seq_length - 1

        if chunk_size <= 0:
            raise ValueError(f"max_seq_length ({max_seq_length}) must be at least 2 to fit BOS + content")

        if total_length < chunk_size:
            return {k: [] for k in examples.keys()}

        # 3. Trim remainder to make it divisible by chunk_size
        total_length = (total_length // chunk_size) * chunk_size

        # 4. Chunk into blocks of (max_seq_length - 1)
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }

        # 5. Add BOS at start of each block
        for i in range(len(result["input_ids"])):
            result["input_ids"][i] = [bos_token_id] + result["input_ids"][i]

            if "attention_mask" in result:
                result["attention_mask"][i] = [1] + result["attention_mask"][i]

        return result

    def _truncate_datasets(self, tokenized_ds: dict[str, Dataset]) -> dict[str, Dataset]:
        """Fallback truncation for when packing is disabled."""
        final_ds = {}
        for split_name, split_data in tokenized_ds.items():
            final_ds[split_name] = split_data.map(
                partial(self._truncate_sequences, max_seq_length=self.max_seq_length),
                batched=True,
                num_proc=max(1, self.num_proc),
                desc=f"Truncating {split_name} to {self.max_seq_length} tokens",
            )
        return final_ds

    @staticmethod
    def _truncate_sequences(examples: dict[str, list], max_seq_length: int) -> dict[str, list]:
        """Fallback truncation for when packing is disabled."""
        result = {}
        for k, sequences in examples.items():
            result[k] = [seq[:max_seq_length] for seq in sequences]
        return result
