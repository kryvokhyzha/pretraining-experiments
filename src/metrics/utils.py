import re

from transformers import PreTrainedTokenizerBase


# Comprehensive pattern to match role markers at the beginning of text
_ROLE_PATTERN = re.compile(
    r"^\s*(?:<start_of_turn>\s*)?(?:<\|?(?:assistant|model|user|system)\|?>\s*:?\s*|"
    r"(?:assistant|model|user|system)\s*:?\s*|"
    r"(?:Assistant|Model|User|System)\s*:?\s*|"
    r"(?:ASSISTANT|MODEL|USER|SYSTEM)\s*:?\s*|"
    r"A:\s*|Assistant:\s*|Model:\s*)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def detect_eot_ids(tokenizer: PreTrainedTokenizerBase) -> list[int] | None:
    """Detect end-of-turn token IDs for chat models.

    Args:
        tokenizer: The tokenizer to inspect

    Returns:
        List of EOT token IDs or None if not found

    """
    eot_ids = []

    # Check for common EOT tokens
    if hasattr(tokenizer, "eot_token_id"):
        eot_ids.append(tokenizer.eot_token_id)

    # Check for special tokens that might be EOT
    candidates = [
        "<|eot_id|>",
        "<|im_end|>",
        "<|endoftext|>",
        "</s>",
        "<end_of_turn>",
        "<|eot_id|>",
        "<eot_id>",
        "<|eot|>",
        "<EOT>",
    ]
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            eot_ids.append(token_id)

    # Also include eos_token_id if it exists
    if tokenizer.eos_token_id is not None:
        eot_ids.append(tokenizer.eos_token_id)

    return list(set(eot_ids)) if eot_ids else None


def strip_role_markers(text: str) -> str:
    """Strip role markers from generated text.

    This function handles cases where:
    1. Role markers appear at the beginning of text
    2. The model echoes the conversation including role headers
    3. Multiple role markers are present

    Args:
        text: The generated text

    Returns:
        Text with role markers stripped

    """
    if not text:
        return text

    # First, check if the model echoed the conversation with role headers
    # Look for cases where "user" appears followed by "model" or "assistant"
    text_lower = text.lower()

    # Find all occurrences of role markers that might indicate echoed conversation
    role_positions = []
    for role in ["model", "assistant"]:
        pos = text_lower.rfind(role)
        if pos >= 0:
            role_positions.append((pos, role))

    # If we found a model/assistant marker after potential user content
    if role_positions:
        # Get the position of the last model/assistant marker
        last_pos, last_role = max(role_positions, key=lambda x: x[0])

        # Check if there's a "user" marker before it
        user_pos = text_lower.rfind("user", 0, last_pos)

        if user_pos >= 0 and user_pos < last_pos:
            # Extract text after the last model/assistant marker
            # Add length of role word plus common separators
            start_idx = last_pos + len(last_role)
            # Skip common separators like ":", whitespace, etc.
            while start_idx < len(text) and text[start_idx] in ": \n\t":
                start_idx += 1

            tail = text[start_idx:].strip()

            # Only use this if it's not too short (avoid extracting fragments)
            # and not too much of the original (avoid false positives)
            if tail and len(tail) >= 10 and len(tail) < len(text) * 0.9:
                return tail

    # Fallback: Remove role markers from the beginning using regex
    cleaned = _ROLE_PATTERN.sub("", text, count=1).strip()

    # Additional cleanup: if text starts with common continuation markers, remove them
    continuation_markers = [":", "-", "•", "*", ">"]
    while cleaned and cleaned[0] in continuation_markers:
        cleaned = cleaned[1:].strip()

    return cleaned
