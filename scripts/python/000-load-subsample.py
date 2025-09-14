from dotenv import find_dotenv, load_dotenv
from huggingface_hub import hf_hub_download


load_dotenv(find_dotenv(), override=True)


if __name__ == "__main__":
    file_path = hf_hub_download(repo_id="Goader/kobza", repo_type="dataset", filename="000_ukr_corpus_00000.parquet")

    print(f"File saved at: {file_path}")
