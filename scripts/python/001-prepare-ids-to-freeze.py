import hydra
import joblib
import rootutils
import torch
from dotenv import find_dotenv, load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer


path_to_root = rootutils.find_root(search_from=__file__, indicator=".project-root")
load_dotenv(find_dotenv(), override=True)


def get_shared_token_ids(tokenizer_a: PreTrainedTokenizer, tokenizer_b: PreTrainedTokenizer) -> list[int]:
    vocab_a = tokenizer_a.get_vocab()
    vocab_b = tokenizer_b.get_vocab()

    inv_a = {v: k for k, v in vocab_a.items()}
    inv_b = {v: k for k, v in vocab_b.items()}

    shared_ids = set(inv_a.keys()) & set(inv_b.keys())

    return [i for i in shared_ids if inv_a[i] == inv_b[i]]


@hydra.main(version_base=None, config_path="../../configs", config_name="prepare_ids")
def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model.main)
    tokenizer_original = instantiate(cfg.tokenizer_original.tokenizer)
    tokenizer_modified = instantiate(cfg.tokenizer_modified.tokenizer)

    ids_same_id_and_same_token = get_shared_token_ids(tokenizer_original, tokenizer_modified)

    t = torch.tensor(sorted(ids_same_id_and_same_token), dtype=torch.long)
    valid_mask = t < model.get_input_embeddings().weight.shape[0]
    ids_to_freeze = t[valid_mask]

    joblib.dump(ids_to_freeze, path_to_root / cfg.path_to_freeze_ids)


if __name__ == "__main__":
    main()
