from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import pandas as pd
import os

seed = 42
MAX_ARTICLE_TOKENS = 1024

def prepare_dataset(n=500):
    path = hf_hub_download(
        repo_id="abisee/cnn_dailymail",
        filename="3.0.0/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    df["n_tokens"] = [len(ids) for ids in tok(df["article"].tolist(), add_special_tokens=False).input_ids]
    print(df["n_tokens"].describe(percentiles=[.5, .9, .95]))
    kept = df[df["n_tokens"] <= MAX_ARTICLE_TOKENS]
    print(f"Kept {len(kept)}/{len(df)} articles with <= {MAX_ARTICLE_TOKENS} tokens")

    sub = kept.sample(n=n, random_state=seed).reset_index(drop=True)
    os.makedirs("datasets", exist_ok=True)
    sub[["id", "article", "highlights"]].to_json("datasets/cnn_dm_eval.jsonl", orient="records", lines=True)
    print(f"Saved {len(sub)} articles")

if __name__ == "__main__":
    prepare_dataset()