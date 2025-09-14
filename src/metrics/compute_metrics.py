import math
from typing import Dict

import torch
from transformers import EvalPrediction


def compute_metrics_perplexity(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute perplexity from evaluation predictions.

    Args:
        eval_pred: EvalPrediction containing predictions and labels

    Returns:
        Dictionary containing perplexity and loss metrics

    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    # predictions are logits, we need to compute cross-entropy loss
    # predictions shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)

    # Convert to tensors if they aren't already
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.from_numpy(predictions)
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels)

    # For causal language modeling, we shift the labels
    # We predict next token, so we compare predictions[i] with labels[i+1]
    shift_logits = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute token-level losses
    losses = loss_fct(shift_logits, shift_labels)

    # Mask out ignored tokens (where label == -100)
    valid_mask = shift_labels != -100
    valid_losses = losses[valid_mask]

    if len(valid_losses) == 0:
        return {"perplexity": float("inf"), "eval_loss": float("inf")}

    # Compute mean loss over valid tokens
    mean_loss = valid_losses.mean().item()

    # Compute perplexity (exp of mean loss)
    try:
        perplexity = math.exp(mean_loss)
        # Cap perplexity at a reasonable maximum to avoid overflow issues
        if perplexity > 1e10:
            perplexity = float("inf")
    except OverflowError:
        perplexity = float("inf")

    return {
        "perplexity": perplexity,
        "eval_loss": mean_loss,
    }
