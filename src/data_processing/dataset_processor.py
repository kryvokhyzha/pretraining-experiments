import os
from typing import Dict, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from src.helper.logging import logger


def is_local_parquet(path: str) -> bool:
    """Check if the path is a local parquet file."""
    return os.path.exists(path) and (path.endswith(".parquet") or path.endswith(".parquet."))


def prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    num_proc: int,
    max_records: Optional[int] = None,
    text_column: str = "text",
    eval_split_percentage: float = 0.1,
    create_eval_split: bool = False,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Load dataset either from local parquet file(s) or from HF dataset id.

    Expects dataset text column named by text_column parameter.
    If create_eval_split is True, creates train/eval splits.

    Args:
        dataset_name: Path to local parquet file or HuggingFace dataset name
        tokenizer: Tokenizer to use for tokenization
        max_seq_length: Maximum sequence length for tokenization
        num_proc: Number of processes for parallel processing
        max_records: Maximum number of records to use (None for all)
        text_column: Name of the text column in the dataset
        eval_split_percentage: Percentage of data to use for evaluation split
        create_eval_split: Whether to create evaluation split if it doesn't exist
        seed: Random seed for reproducible splits

    Returns:
        Dictionary containing tokenized datasets splits

    """
    # Step 1: Load the dataset
    ds = _load_raw_dataset(dataset_name=dataset_name, create_eval_split=create_eval_split)

    # Step 2: Create evaluation split if needed (before limiting records)
    if create_eval_split and "validation" not in ds:
        ds = _create_train_eval_split(ds=ds, eval_split_percentage=eval_split_percentage, seed=seed)

    # Step 3: Limit records if specified (after splitting)
    if max_records is not None and max_records > 0:
        ds = _limit_dataset_records(ds=ds, max_records=max_records, eval_split_percentage=eval_split_percentage)

    # Step 4: Ensure text column exists and tokenize
    text_column = _validate_text_column(dataset=ds["train"], text_column=text_column)
    tokenized_ds = _tokenize_datasets(
        ds=ds, tokenizer=tokenizer, text_column=text_column, max_seq_length=max_seq_length, num_proc=num_proc
    )

    return tokenized_ds


def _load_raw_dataset(dataset_name: str, create_eval_split: bool) -> Dict[str, Dataset]:
    """Load dataset from local parquet or HuggingFace hub."""
    if is_local_parquet(path=dataset_name):
        ds = load_dataset("parquet", data_files=dataset_name, split="train")
        return {"train": ds}

    try:
        # Try to load with existing splits
        ds = load_dataset(dataset_name)
        if "validation" not in ds and "test" not in ds and create_eval_split:
            # No validation split exists, we'll create one from train
            return {"train": ds["train"]}
        else:
            # Has validation or test split, use as is
            logger.info(f"Dataset has existing splits: {list(ds.keys())}")
            return {"train": ds["train"]} if not create_eval_split else ds
    except Exception:
        # Fallback to loading just train split
        ds = load_dataset(dataset_name, split="train")
        return {"train": ds}


def _create_train_eval_split(ds: Dict[str, Dataset], eval_split_percentage: float, seed: int) -> Dict[str, Dataset]:
    """Create train/eval split from the train dataset."""
    train_size = len(ds["train"])

    if train_size <= 1:
        logger.warning(
            f"Cannot create eval split: dataset has only {train_size} record(s). "
            "Using train dataset for both training and evaluation."
        )
        ds["validation"] = ds["train"]
        return ds

    logger.info(f"Creating train/eval split with {eval_split_percentage:.1%} for evaluation")

    eval_size = max(1, int(train_size * eval_split_percentage))
    split_ds = ds["train"].train_test_split(
        test_size=eval_size,
        shuffle=True,
        seed=seed,
    )

    ds["train"] = split_ds["train"]
    ds["validation"] = split_ds["test"]

    logger.info(f"Split created - Train: {len(ds['train'])}, Eval: {len(ds['validation'])}")
    return ds


def _limit_dataset_records(
    ds: Dict[str, Dataset], max_records: int, eval_split_percentage: float
) -> Dict[str, Dataset]:
    """Limit the number of records in each split."""
    for split_name in list(ds.keys()):
        original_size = len(ds[split_name])

        if split_name == "train" and "validation" in ds:
            # For training, use most of the max_records
            train_records = max(1, int(max_records * (1 - eval_split_percentage)))
            limit = min(train_records, original_size)
        elif split_name == "validation" and "train" in ds:
            # For validation, use remaining records
            eval_records = max(1, max_records - len(ds["train"]))
            limit = min(eval_records, original_size)
        else:
            # For other splits or single split, use max_records
            limit = min(max_records, original_size)

        ds[split_name] = ds[split_name].select(range(limit))

    splits_info = ", ".join([f"{name}: {len(split)}" for name, split in ds.items()])
    logger.info(f"Dataset limited - {splits_info}")
    return ds


def _validate_text_column(dataset: Dataset, text_column: str) -> str:
    """Validate that the text column exists in the dataset."""
    if text_column in dataset.column_names:
        return text_column

    # Fallback to search for text-like columns
    for col in dataset.column_names:
        if col.lower() in ("text", "document", "content"):
            logger.info(f"Using column '{col}' as text column instead of '{text_column}'")
            return col

    raise ValueError(f"No text-like column found in dataset. Columns: {dataset.column_names}")


def _tokenize_datasets(
    ds: Dict[str, Dataset], tokenizer: AutoTokenizer, text_column: str, max_seq_length: int, num_proc: int
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
