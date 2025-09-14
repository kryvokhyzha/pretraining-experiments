import joblib
import rootutils
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


path_to_root = rootutils.find_root(search_from=__file__, indicator=".project-root")


def get_shared_token_ids(tokenizer_a: AutoTokenizer, tokenizer_b: AutoTokenizer) -> list[int]:
    vocab_a = tokenizer_a.get_vocab()
    vocab_b = tokenizer_b.get_vocab()

    inv_a = {v: k for k, v in vocab_a.items()}
    inv_b = {v: k for k, v in vocab_b.items()}

    shared_ids = set(inv_a.keys()) & set(inv_b.keys())

    return [i for i in shared_ids if inv_a[i] == inv_b[i]]


def main() -> None:
    model_gemma = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")
    tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    tokenizer_modified = AutoTokenizer.from_pretrained("transhumanist-already-exists/tereshchenkoblue-tokenizer")

    ids_same_id_and_same_token = get_shared_token_ids(tokenizer_gemma, tokenizer_modified)

    t = torch.tensor(sorted(ids_same_id_and_same_token), dtype=torch.long)
    valid_mask = t < model_gemma.get_input_embeddings().weight.shape[0]
    ids_to_freeze = t[valid_mask]

    joblib.dump(ids_to_freeze, path_to_root / "data" / "freeze_ids_gemma_tereshchenko.joblib")


if __name__ == "__main__":
    main()
