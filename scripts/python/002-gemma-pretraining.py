import os
from functools import partial

import hydra
import joblib
import rootutils
import torch
from datasets import Dataset
from dotenv import find_dotenv, load_dotenv
from hydra.core.hydra_config import HydraConfig
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

from src.data_processing.dataset_processor import PretrainingDatasetProcessor
from src.helper.display import DisplayConsole
from src.helper.logging import logger
from src.metrics import compute_metrics_perplexity


# -------------------------
# Utilities
# -------------------------
def create_training_arguments(
    cfg: DictConfig,
    project_name: str,
    report_to: str,
    torch_dtype: torch.dtype,
) -> TrainingArguments:
    """Create TrainingArguments from Hydra config using instantiate."""
    training_config = OmegaConf.to_container(cfg.pretraining.pretrain_config, resolve=True)
    training_config["report_to"] = report_to

    training_config["fp16"] = True if torch_dtype == torch.float16 else False
    training_config["bf16"] = True if torch_dtype == torch.bfloat16 else False

    # Get Hydra runtime configuration
    hydra_cfg = HydraConfig.get()

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

    # Extract model choice from Hydra runtime
    model_choice = hydra_cfg.runtime.choices.get("model", "unknown-model")

    # Extract pretraining type from Hydra runtime
    pretraining_choice = hydra_cfg.runtime.choices.get("pretraining", "unknown")

    # Construct dynamic output_dir and run_name
    run_name = f"{project_name}-{model_choice}-{pretraining_choice}".replace("_", "-")
    output_dir = f"./outputs/checkpoints/{run_name}"

    # Override the config values
    training_config["run_name"] = run_name
    training_config["output_dir"] = output_dir

    logger.info(f"Model choice: {model_choice}")
    logger.info(f"Pretraining type: {pretraining_choice}")
    logger.info(f"Setting run_name: {run_name}")
    logger.info(f"Setting output_dir: {output_dir}")

    return instantiate(training_config)


def _hook(grad: torch.tensor, ids: torch.tensor) -> torch.Tensor | None:
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
    console = DisplayConsole()

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
    experiment_tracker = instantiate(cfg.experiment_tracking)

    # Setup experiment tracking
    project_name: str = os.getenv("PROJECT_NAME")
    if project_name is None:
        raise ValueError("PROJECT_NAME environment variable is not set.")
    config_dict: dict = OmegaConf.to_container(cfg, resolve=True)
    experiment_tracker.init(project=project_name, config=config_dict)

    # Tokenizer + model using Hydra instantiate
    logger.info("Loading tokenizer...")
    tokenizer = instantiate(cfg.tokenizer.tokenizer)

    logger.info("Tokenizer default settings:")
    logger.info(f"  - pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"  - padding_side: {tokenizer.padding_side}")
    logger.info(f"  - truncation_side: {tokenizer.truncation_side}")

    if cfg.tokenizer.get("kwargs") is not None:
        logger.info("Applying tokenizer kwargs...")
        tokenizer_kwargs = OmegaConf.to_container(cfg.tokenizer.kwargs, resolve=True)

        if tokenizer_kwargs.get("padding_side") is not None:
            tokenizer.padding_side = tokenizer_kwargs["padding_side"]

        if tokenizer_kwargs.get("truncation_side") is not None:
            tokenizer.truncation_side = tokenizer_kwargs["truncation_side"]

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Log final tokenizer settings
    logger.info("Tokenizer final settings:")
    logger.info(f"  - pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"  - padding_side: {tokenizer.padding_side}")
    logger.info(f"  - truncation_side: {tokenizer.truncation_side}")
    logger.info(f"  - bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"  - eos_token_id: {tokenizer.eos_token_id}")

    logger.info("Loading model...")
    model = instantiate(cfg.model.main)
    logger.info(f"Using model on device: {model.device}")

    # # If tokenizer extended (pad) we must resize token embeddings
    # model.resize_token_embeddings(len(tokenizer))

    if cfg.get("path_to_freeze_ids", None) is not None:
        logger.info("Freezing specified token embeddings...")
        ids_to_freeze = joblib.load(cfg.path_to_freeze_ids)
        _ = model.get_input_embeddings().weight.register_hook(partial(_hook, ids=ids_to_freeze))
        logger.info(f"Number of frozen token embeddings: {len(ids_to_freeze)}")

    # Prepare dataset using the new class-based processor
    logger.info("Loading and tokenizing dataset...")

    dataset_processor = PretrainingDatasetProcessor(
        tokenizer=tokenizer,
        max_seq_length=cfg.data_processing.max_seq_length,
        text_column=cfg.data.text_column,
        num_proc=cfg.data_processing.num_proc,
        use_packing=cfg.data_processing.use_packing,
        seed=cfg.seed,
    )

    datasets = dataset_processor.prepare_dataset(
        path=cfg.data.path,
        n_train=cfg.data_processing.n_train,
        n_val=cfg.data_processing.n_val,
        n_test=cfg.data_processing.n_test,
    )

    # Extract train and eval datasets
    train_dataset: Dataset = datasets["train"]
    eval_dataset: Dataset | None = datasets.get("validation", None)

    console.display_df_as_table(train_dataset.to_pandas()[:2], title="Example of Train Data", max_col_width=None)
    if eval_dataset:
        console.display_df_as_table(eval_dataset.to_pandas()[:2], title="Example of Eval Data", max_col_width=None)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, seed=cfg.seed)

    # Create training arguments from config
    training_args: TrainingArguments = create_training_arguments(
        cfg,
        project_name=project_name,
        report_to=experiment_tracker.get_report_to_string(),
        torch_dtype=model.dtype,
    )

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

    if training_args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            model.gradient_checkpointing = True
        model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics_perplexity,
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
