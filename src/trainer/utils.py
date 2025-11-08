import torch

from src.helper.logging import logger


def create_embedding_freeze_hook(token_ids: list[int]) -> callable:
    """Create a hook that zeros gradients for specific token IDs.

    Args:
        token_ids: List of token IDs whose embeddings should be frozen

    Returns:
        Hook function that zeros gradients for specified tokens

    """
    # Convert to tensor and sort for efficiency
    token_ids_tensor = torch.tensor(sorted(set(token_ids)), dtype=torch.long)

    def hook(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return None

        device_ids = token_ids_tensor.to(grad.device)
        modified_grad = grad.clone()
        modified_grad[device_ids] = 0

        return modified_grad

    return hook


def install_freeze_hooks(
    model: torch.nn.Module,
    token_ids: list[int],
    include_lm_head: bool = True,
) -> None:
    """Install gradient hooks to freeze specific token embeddings.

    Args:
        model: The model to install hooks on
        token_ids: List of token IDs to freeze
        include_lm_head: Whether to also freeze the output projection layer

    """
    hook_fn = create_embedding_freeze_hook(token_ids)

    # Install hook on input embeddings
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        _ = input_embeddings.weight.register_hook(hook_fn)
        logger.info(f"Installed freeze hook on input embeddings for {len(token_ids)} tokens")

    # Install hook on output embeddings (lm_head)
    if include_lm_head:
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            _ = output_embeddings.weight.register_hook(hook_fn)
            logger.info(f"Installed freeze hook on output embeddings for {len(token_ids)} tokens")
