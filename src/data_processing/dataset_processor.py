import os
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

    # Step 4: Ensure text column exists and tokenize
    tokenized_ds = _tokenize_datasets(
        ds=ds,
        tokenizer=tokenizer,
        text_column=text_column,
        max_seq_length=max_seq_length,
        num_proc=num_proc,
    )

    return tokenized_ds


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
    max_seq_length: int,
    num_proc: int,
) -> Dict[str, Dataset]:
    """Tokenize all datasets splits."""

    def tokenize_batch(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_seq_length)

    tokenized_ds = {}
    for split_name, split_data in ds.items():
        tokenized_ds[split_name] = split_data.map(
            tokenize_batch,
            batched=True,
            num_proc=max(1, num_proc),
            remove_columns=split_data.column_names,
        )

    return tokenized_ds
