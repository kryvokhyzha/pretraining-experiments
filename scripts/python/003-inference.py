import time

import hydra
import rootutils
import torch
from dotenv import find_dotenv, load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


_ = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=False)
load_dotenv(find_dotenv(), override=True)

from src.helper.display import DisplayConsole
from src.helper.logging import logger
from src.metrics.utils import detect_eot_ids


@hydra.main(version_base=None, config_path="../../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    console = DisplayConsole()

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = instantiate(OmegaConf.to_container(cfg.tokenizer.tokenizer, resolve=True))

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

    # Ensure tokenizer has pad token for collator
    logger.info("Tokenizer final settings:")
    logger.info(f"  - pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"  - padding_side: {tokenizer.padding_side}")
    logger.info(f"  - truncation_side: {tokenizer.truncation_side}")
    logger.info(f"  - bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"  - eos_token_id: {tokenizer.eos_token_id}")
    logger.info(f"  - pad_token_id: {tokenizer.pad_token_id}")

    eos_ids = detect_eot_ids(tokenizer)
    if eos_ids:
        logger.info(f"  - detected eos token IDs: {eos_ids}")

    # Load model
    logger.info("Loading model...")
    model = instantiate(cfg.model.main)

    model.eval()
    device = model.device
    logger.info(f"Model loaded on device: {device}")

    # Prepare generation config
    generation_config = OmegaConf.to_container(cfg.generation, resolve=True)
    logger.info(f"Generation config: {generation_config}")

    # Generate completions
    logger.info(f"Generating completions for {len(cfg.prompts)} prompts...")

    # Pause for 2 seconds before starting
    time.sleep(2)

    for idx, prompt in enumerate(cfg.prompts, 1):
        console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
        console.print(f"[bold]Prompt {idx}/{len(cfg.prompts)}:[/bold] {prompt}")
        console.print(f"[bold cyan]{'=' * 80}[/bold cyan]")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids or tokenizer.eos_token_id,
            )

        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, text in enumerate(generated_texts):
            console.print(f"\n[bold green]Completion {i + 1}:[/bold green]")
            console.print(text)

    # Pause for 1 second before finishing
    time.sleep(1)
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
