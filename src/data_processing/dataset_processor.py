import os
from functools import partial
from typing import Dict, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from src.helper.logging import logger


def is_local_parquet(path: str) -> bool:
    """Check if the path is a local parquet file."""
    return os.path.exists(path) and (path.endswith(".parquet") or path.endswith(".parquet."))


def prepare_dataset(
    path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    num_proc: int,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    text_column: str = "text",
    seed: int = 42,
    use_packing: bool = True,
) -> Dict[str, Dataset]:
    """Load dataset either from local parquet file(s) or from HF dataset id.

    Expects dataset text column named by text_column parameter.
    If create_eval_split is True, creates train/eval splits.

    Args:
        path: Path to local parquet file or HuggingFace dataset name
        tokenizer: Tokenizer to use for tokenization
        max_seq_length: Maximum sequence length for tokenization
        num_proc: Number of processes for parallel processing
        n_train: Number of training records to use (None for all)
        n_val: Number of validation records to use (None for all)
        n_test: Number of test records to use (None for all)
        text_column: Name of the text column in the dataset
        seed: Random seed for reproducible splits
        use_packing: Whether to use packing (concatenation + chunking) for efficiency

    Returns:
        Dictionary containing tokenized datasets splits

    """
    # Step 1: Load the dataset
    ds = _load_raw_dataset(path=path)

    # Step 2: Create evaluation split if needed (before limiting records)
    if n_val or n_test:
        ds = _create_train_eval_split(ds=ds, n_val=n_val, n_test=n_test, seed=seed)

    # Step 3: Limit records if specified (after splitting)
    if n_train is not None and n_train > 0:
        ds = _limit_dataset_records(ds=ds, n_train=n_train, n_val=n_val, n_test=n_test)

    # Step 4: Tokenize with EOS separator (without special tokens)
    tokenized_ds = _tokenize_datasets(
        ds=ds,
        tokenizer=tokenizer,
        text_column=text_column,
        num_proc=num_proc,
    )

    # Step 5: Apply packing if enabled
    if use_packing:
        logger.info(f"Applying packing to group sequences into {max_seq_length} token blocks...")

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        if bos_id is None:
            raise ValueError("Tokenizer must have a BOS token defined for packing.")
        if eos_id is None:
            raise ValueError("Tokenizer must have an EOS token defined for packing.")

        packed_ds = {}
        for split_name, split_data in tokenized_ds.items():
            packed_ds[split_name] = split_data.map(
                partial(
                    _group_texts,
                    max_seq_length=max_seq_length,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                ),
                batched=True,
                num_proc=max(1, num_proc),
                desc=f"Packing {split_name} into {max_seq_length}-token blocks",
            )
            logger.info(f"Packed {split_name}: {len(packed_ds[split_name])} blocks of {max_seq_length} tokens")

        return packed_ds
    else:
        # Fallback: traditional truncation (less efficient)
        logger.warning("Packing disabled - using truncation (less efficient for pretraining)")
        final_ds = {}
        for split_name, split_data in tokenized_ds.items():
            final_ds[split_name] = split_data.map(
                partial(_truncate_sequences, max_seq_length=max_seq_length),
                batched=True,
                num_proc=max(1, num_proc),
                desc=f"Truncating {split_name} to {max_seq_length} tokens",
            )
        return final_ds


def _load_raw_dataset(path: str) -> Dict[str, Dataset]:
    """Load dataset from local parquet or HuggingFace hub."""
    if is_local_parquet(path=path):
        ds = load_dataset("parquet", data_files=path, split="train")
        return {"train": ds}
    else:
        return load_dataset(path=path)


def _create_train_eval_split(
    ds: Dict[str, Dataset],
    n_val: Optional[int],
    n_test: Optional[int],
    seed: int,
) -> Dict[str, Dataset]:
    """Create train/eval split from the train dataset."""
    if "train" not in ds:
        logger.warning("No 'train' split found in dataset. Returning dataset as-is.")
        return ds

    # Shuffle the dataset before splitting
    ds["train"] = ds["train"].shuffle(seed=seed)

    train_size = len(ds["train"])

    # Calculate total records needed for splits
    total_split_size = (n_val or 0) + (n_test or 0)

    if total_split_size == 0:
        # No splits requested
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
            seed=seed,
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
            seed=seed,
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


def _limit_dataset_records(
    ds: Dict[str, Dataset],
    n_train: Optional[int],
    n_val: Optional[int],
    n_test: Optional[int],
) -> Dict[str, Dataset]:
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


def _tokenize_datasets(
    ds: Dict[str, Dataset],
    tokenizer: AutoTokenizer,
    text_column: str,
    num_proc: int,
) -> Dict[str, Dataset]:
    """Tokenize all datasets splits and add EOS separator between documents.

    This tokenization step:
    1. Disables automatic special token addition (add_special_tokens=False)
    2. Adds EOS token manually after each document as separator
    3. Does NOT truncate (truncation will happen during packing)
    """

    def tokenize_batch(examples):
        # Tokenize without automatic special tokens
        tokenized_output = tokenizer(
            examples[text_column],
            add_special_tokens=False,
            truncation=False,  # Don't truncate yet - packing will handle this
        )

        # Add EOS as separator after each document
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must have an EOS token defined.")

        tokenized_output["input_ids"] = [ids + [eos_id] for ids in tokenized_output["input_ids"]]

        # Handle attention_mask if present
        if "attention_mask" in tokenized_output:
            tokenized_output["attention_mask"] = [mask + [1] for mask in tokenized_output["attention_mask"]]

        return tokenized_output

    tokenized_ds = {}
    for split_name, split_data in ds.items():
        tokenized_ds[split_name] = split_data.map(
            tokenize_batch,
            batched=True,
            num_proc=max(1, num_proc),
            remove_columns=split_data.column_names,
            desc=f"Tokenizing and adding EOS for {split_name}",
        )

    return tokenized_ds


def _group_texts(
    examples: Dict[str, list],
    max_seq_length: int,
    bos_token_id: int,
    eos_token_id: int,
) -> Dict[str, list]:
    """Concatenate all sequences into one long list and chunk into fixed-length blocks.

    This implements packing for efficient pretraining:
    1. Concatenates all tokenized documents (each ending with EOS from _tokenize_datasets)
    2. Chunks into blocks of (max_seq_length - 2) to reserve space for BOS and final EOS
    3. Adds BOS at start and EOS at end of each block

    Final structure of each block:
    [<bos>, doc1_tokens..., <eos>, doc2_tokens..., <eos>]

    Note: Documents within the block already have EOS separators from tokenization,
    and we add a final EOS at the end of each block.

    Args:
        examples: Batch of tokenized examples (each document ends with EOS)
        max_seq_length: Target sequence length for each block (including BOS and final EOS)
        bos_token_id: BOS token ID to prepend at start of each block
        eos_token_id: EOS token ID to append at end of each block

    Returns:
        Dictionary with packed sequences

    """
    # 1. Concatenate all texts into one super-long list
    # Each document already has EOS at the end from _tokenize_datasets
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # 2. Calculate chunk size (reserve 1 token for BOS, 1 for final EOS)
    chunk_size = max_seq_length - 2

    if chunk_size <= 0:
        raise ValueError(f"max_seq_length ({max_seq_length}) must be at least 3 to fit BOS + content + EOS")

    if total_length < chunk_size:
        # If we don't have enough tokens for even one block, return empty
        return {k: [] for k in examples.keys()}

    # 3. Trim remainder to make it divisible by chunk_size
    total_length = (total_length // chunk_size) * chunk_size

    # 4. Chunk into blocks of (max_seq_length - 2) to leave room for BOS and final EOS
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)] for k, t in concatenated_examples.items()
    }

    # 5. CRITICAL FOR GEMMA: Add BOS at start and EOS at end of each block
    # This ensures: [<bos>, content_with_internal_eos..., <eos>]
    for i in range(len(result["input_ids"])):
        # Prepend BOS and append EOS to make final length = max_seq_length
        result["input_ids"][i] = [bos_token_id] + result["input_ids"][i] + [eos_token_id]

        # Update attention_mask if it exists
        if "attention_mask" in result:
            result["attention_mask"][i] = [1] + result["attention_mask"][i] + [1]

    return result


def _truncate_sequences(
    examples: Dict[str, list],
    max_seq_length: int,
) -> Dict[str, list]:
    """Fallback truncation for when packing is disabled."""
    result = {}
    for k, sequences in examples.items():
        result[k] = [seq[:max_seq_length] for seq in sequences]
    return result
