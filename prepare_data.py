from datasets import load_dataset
import pandas as pd
import os

# In this file, we construct the combining training and test datasets
seed = 42

def prepare_gsm8k(split):
    df = load_dataset("openai/gsm8k", "main", split=split).to_pandas()
    parts = df["answer"].str.split("####", n=1, expand=True)
    rationale = parts[0].str.replace(r"<<.*?>>", "", regex=True).str.strip()
    answer = parts[1].str.strip().str.replace(",", "", regex=False)
    return pd.DataFrame({
        "id": [f"gsm8k-{split}-{i}" for i in range(len(df))],
        "source": "gsm8k",
        "question": df["question"],
        "solution": rationale + "\nTherefore, the answer is $\\boxed{" + answer + "}$.",
        "answer": answer,
        "level": pd.NA,
        "subject": pd.NA,
    })

def prepare_math(split):
    df = load_dataset("nlile/hendrycks-MATH-benchmark", split=split).to_pandas()
    return pd.DataFrame({
        "id": df["unique_id"],
        "source": "math",
        "question": df["problem"],
        "solution": df["solution"],
        "answer": df["answer"],
        "level": df["level"],
        "subject": df["subject"],
    })

def split_val(df, val_frac=0.05):
    val_idx = df.sample(frac=val_frac, random_state=seed).index
    return df.drop(val_idx), df.loc[val_idx]

if __name__ == '__main__':
    # Prepare the datasets
    gsm8k_train, gsm8k_test = prepare_gsm8k("train"), prepare_gsm8k("test")
    gsm8k_train, gsm8k_val = split_val(gsm8k_train, val_frac=0.1)
    math_train, math_500 = prepare_math("train"), prepare_math("test")
    math_train, math_val = split_val(math_train, val_frac=0.1)
    val = pd.concat([gsm8k_val, math_val], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    train = pd.concat([gsm8k_train, math_train], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    os.makedirs("datasets", exist_ok=True)
    for name, d in [("train", train), ("val", val), ("gsm8k_test", gsm8k_test), ("math_500", math_500)]:
        d.to_json(f"datasets/{name}.jsonl", orient="records", lines=True)
        print(name, len(d), d["source"].value_counts().to_dict())