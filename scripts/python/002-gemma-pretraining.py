import os
from functools import partial
from typing import Dict, Optional

import hydra
import joblib
import rootutils
import torch
from datasets import Dataset
from dotenv import find_dotenv, load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False


_ = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=False)
load_dotenv(find_dotenv(), override=True)

from src.data_processing import prepare_dataset
from src.helper.logging import logger
from src.metrics import compute_metrics_perplexity


# -------------------------
# Utilities
# -------------------------
def create_training_arguments(cfg: DictConfig, report_to: str) -> TrainingArguments:
    """Create TrainingArguments from Hydra config using instantiate."""
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    training_config["report_to"] = report_to

    # TPU-specific optimizations
    if TPU_AVAILABLE and xm.xla_device_hw(xm.xla_device()) == "TPU":
        # TPU works best with bf16
        training_config["bf16"] = True
        training_config["fp16"] = False
        # TPU-optimized batch sizes (should be divisible by 8)
        if training_config.get("per_device_train_batch_size", 1) % 8 != 0:
            logger.warning("TPU works best with batch sizes divisible by 8")
        # Disable tqdm for better TPU performance
        training_config["disable_tqdm"] = True
        # Use XLA-optimized dataloader
        training_config["dataloader_drop_last"] = True

    return instantiate(training_config)


def _hook(grad: torch.tensor, ids: torch.tensor) -> Optional[torch.tensor]:
    if grad is None:
        return None

    # Make a copy of the gradient tensor to avoid in-place modification issues
    # But probably this is not necessary
    grad = grad.clone()
    grad[ids] = 0
    return grad


# -------------------------
# Main
# -------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Detect device type
    if TPU_AVAILABLE and xm.xla_device_hw(xm.xla_device()) == "TPU":
        device_type = "TPU"
        logger.info(f"Running on TPU: {xm.xla_device()}")
    elif torch.cuda.is_available():
        device_type = "GPU"
        logger.info(f"Running on GPU: {torch.cuda.get_device_name()}")
    else:
        device_type = "CPU"
        logger.info("Running on CPU")

    # Initialize experiment tracker
    experiment_tracker = instantiate(cfg.experiment)

    # Setup experiment tracking
    project: str = os.getenv("PROJECT_NAME", "gemma-pretraining")
    config_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    experiment_tracker.init(project=project, config=config_dict)

    # Tokenizer + model using Hydra instantiate
    logger.info("Loading tokenizer...")
    tokenizer = instantiate(cfg.tokenizer)

    # Ensure tokenizer has pad token for collator
    logger.info(f"Tokenizer pad token id: {tokenizer.pad_token_id}")

    logger.info("Loading model...")
    model = instantiate(cfg.model)

    # # If tokenizer extended (pad) we must resize token embeddings
    # model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Freeze specified token embeddings
    if cfg.get("path_to_freeze_ids", None) is not None:
        logger.info("Freezing specified token embeddings...")
        ids_to_freeze = joblib.load(cfg.path_to_freeze_ids)
        _ = model.get_input_embeddings().weight.register_hook(partial(_hook, ids=ids_to_freeze))
        logger.info(f"Number of frozen token embeddings: {len(ids_to_freeze)}")

    # Prepare dataset
    logger.info("Loading and tokenizing dataset...")

    # Check if we need evaluation dataset
    need_eval_dataset: bool = cfg.training.get("eval_strategy") == "steps"
    eval_split_percentage: float = cfg.data_processing.get("eval_split_percentage", 0.1)

    datasets = prepare_dataset(
        dataset_name=cfg.data.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=cfg.data_processing.max_seq_length,
        num_proc=cfg.data_processing.num_proc,
        max_records=cfg.data_processing.max_records,
        text_column=cfg.data.text_column,
        eval_split_percentage=eval_split_percentage,
        create_eval_split=need_eval_dataset,
        seed=cfg.seed,
    )

    # Extract train and eval datasets
    train_dataset: Dataset = datasets["train"]
    eval_dataset: Optional[Dataset] = datasets.get("validation", None)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, seed=cfg.seed)

    # Create training arguments from config
    training_args: TrainingArguments = create_training_arguments(cfg, experiment_tracker.get_report_to_string())

    if device_type == "GPU" and training_args.bf16:
        # prefer fp16 for NVIDIA GPUs
        training_args.fp16 = True
        training_args.bf16 = False
    elif device_type == "TPU":
        # TPU prefers bf16
        training_args.bf16 = True
        training_args.fp16 = False
        # TPU-specific settings
        training_args.tpu_metrics_debug = False
        training_args.past_index = -1  # Disable past key values for memory efficiency

    # Get callbacks from experiment tracker
    callbacks = experiment_tracker.get_callbacks()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        # compute_metrics=compute_metrics_perplexity,
    )

    # Run training
    train_result = trainer.train()

    # Save final model and push logs
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics(split="train", metrics=metrics)
    trainer.save_metrics(split="train", metrics=metrics)
    trainer.save_state()

    # Finish experiment tracking
    experiment_tracker.finish()


if __name__ == "__main__":
    main()
