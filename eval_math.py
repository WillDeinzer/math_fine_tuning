import argparse
import json
import os
import re
import shutil
import tempfile
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:  # vLLM is optional: --backend hf works without it
    LLM = SamplingParams = None

from common import build_prompt, STOP_STRINGS, cut_at_stop, extract_answer, answers_match
from config import MODEL_NAME, SEED, TEST_PATHS as DATASETS, TRAIN_PATH as FEW_SHOT_POOL

FEW_SHOT_SOURCE = {"gsm8k": "gsm8k", "math500": "math"}
FEW_SHOT_MAX_CHARS = 600


def pick_few_shot_examples(source, k):
    pool = pd.read_json(FEW_SHOT_POOL, lines=True, dtype={"answer": str})
    pool = pool[(pool["source"] == source) & (pool["solution"].str.len() <= FEW_SHOT_MAX_CHARS)]
    pool = pool[pool["solution"].str.contains(r"\\boxed", regex=True)]
    return pool.sample(n=k, random_state=SEED)


def make_prompts(questions, shots_df=None):
    prefix = ""
    if shots_df is not None:
        prefix = "".join(f"{build_prompt(q)} {s}\n\n" for q, s in zip(shots_df["question"], shots_df["solution"]))
    return [prefix + build_prompt(q) for q in questions]


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text):
    matches = NUMBER.findall(cut_at_stop(text))
    return matches[-1].replace(",", "") if matches else None


def grade(output, gold):
    boxed = extract_answer(output)
    strict = answers_match(boxed, gold)
    if boxed is not None:
        fallback, lenient = None, strict
    else:
        fallback = last_number(output)
        lenient = answers_match(fallback, gold)
    return {"extracted_answer": boxed, "fallback_answer": fallback, "correct": strict, "correct_lenient": lenient}


def find_init_model(checkpoint):
    for folder in (checkpoint, os.path.dirname(os.path.normpath(checkpoint))):
        path = os.path.join(folder, "run_info.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("init_model")
    return None


def build_merged_model(checkpoint, init_model, dtype_name):
    for adapter in filter(None, (init_model, checkpoint)):
        if not os.path.exists(os.path.join(adapter, "adapter_config.json")):
            raise ValueError(f"{adapter} has no adapter_config.json; point at a checkpoint-N folder")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=getattr(torch, dtype_name), device_map="cpu")
    for adapter in filter(None, (init_model, checkpoint)):
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        print(f"Merged adapter {adapter}")

    out = tempfile.mkdtemp(prefix="merged_model_", dir=os.environ.get("TMPDIR"))
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(out)
    del model
    return out


class VLLMGenerator:
    def __init__(self, model_path, dtype_name, max_model_len, gpu_memory_utilization):
        if LLM is None:
            raise ImportError("vllm is not installed; install it or use --backend hf")
        self.llm = LLM(model=model_path, dtype=dtype_name, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, seed=SEED)

    def generate(self, prompts, max_new_tokens):
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=STOP_STRINGS)
        outputs = self.llm.generate(prompts, params)
        return [(o.outputs[0].text, o.outputs[0].finish_reason == "length") for o in outputs]


class HFGenerator:
    def __init__(self, model_path, dtype_name, batch_size):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype_name),
                                                          device_map="auto").eval()
        self.batch_size = batch_size

    def generate(self, prompts, max_new_tokens):
        results = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tok(prompts[i:i + self.batch_size], return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          stop_strings=STOP_STRINGS, tokenizer=self.tok,
                                          pad_token_id=self.tok.pad_token_id)
            new_tokens = out[:, enc.input_ids.shape[1]:]
            for row in new_tokens:
                text = self.tok.decode(row, skip_special_tokens=True)
                n = int((row != self.tok.pad_token_id).sum())
                stopped = any(s in text for s in STOP_STRINGS) or self.tok.eos_token_id in row.tolist()
                results.append((cut_at_stop(text), n >= max_new_tokens and not stopped))
            print(f"  generated {min(i + self.batch_size, len(prompts))}/{len(prompts)}", flush=True)
        return results


def summarize(df):
    # n: number of problems evaluated
    # accuracy: share whose last \boxed{} answer matches the gold answer (headline metric, same rule as GRPO reward)
    # accuracy_lenient: same, but with no \boxed{} the last number in the output is used instead
    # no_boxed_rate: share of outputs with no \boxed{} answer (format not followed)
    # truncated_rate: share of outputs cut off at max_new_tokens (if high, raise --max_new_tokens)
    # mean_output_chars: average output length in characters
    # accuracy_by_level / accuracy_by_subject: accuracy per MATH difficulty level (1-5) and subject (MATH-500 only)
    m = {
        "n": len(df),
        "accuracy": df["correct"].mean(),
        "accuracy_lenient": df["correct_lenient"].mean(),
        "no_boxed_rate": df["extracted_answer"].isna().mean(),
        "truncated_rate": df["truncated"].mean(),
        "mean_output_chars": df["output"].str.len().mean(),
    }
    if df["level"].notna().any():
        m["accuracy_by_level"] = df.groupby("level")["correct"].mean().sort_index().to_dict()
        m["accuracy_by_subject"] = df.groupby("subject")["correct"].mean().sort_index().to_dict()
    return m


def to_builtin(x):
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if hasattr(x, "item"):
        return x.item()
    return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="LoRA checkpoint-N folder. Omit to evaluate the base model.")
    p.add_argument("--init_model", default=None,
                   help="SFT adapter the checkpoint was trained on top of (default: read from run_info.json)")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--shots", type=int, default=0, help="Number of solved examples in the prompt (0 = zero-shot)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N problems per dataset (for quick tests)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=4096, help="vLLM context window (prompt + output)")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--batch_size", type=int, default=16, help="HF backend only")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM only")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    init_model = None
    if args.checkpoint is not None:
        init_model = args.init_model or find_init_model(args.checkpoint)
    print(f"base: {MODEL_NAME} | init_model: {init_model} | checkpoint: {args.checkpoint} | shots: {args.shots}")

    merged_dir = None
    model_path = MODEL_NAME
    if args.checkpoint is not None:
        merged_dir = build_merged_model(args.checkpoint, init_model, args.dtype)
        model_path = merged_dir

    try:
        if args.backend == "vllm":
            gen = VLLMGenerator(model_path, args.dtype, args.max_model_len, args.gpu_memory_utilization)
        else:
            gen = HFGenerator(model_path, args.dtype, args.batch_size)

        metrics = {}
        for name in args.datasets:
            df = pd.read_json(DATASETS[name], lines=True, dtype={"answer": str})
            if args.limit:
                df = df.head(args.limit)
            shots_df = pick_few_shot_examples(FEW_SHOT_SOURCE[name], args.shots) if args.shots else None
            prompts = make_prompts(df["question"], shots_df)
            print(f"\n=== {name}: {len(df)} problems ===\nexample prompt (last 400 chars):\n{prompts[0][-400:]}")

            t = time.time()
            outputs = gen.generate(prompts, args.max_new_tokens)
            elapsed = time.time() - t

            graded = [grade(text, gold) for (text, _), gold in zip(outputs, df["answer"])]
            out_df = pd.DataFrame({
                "id": df["id"],
                "source": df["source"],
                "level": df["level"],
                "subject": df["subject"],
                "question": df["question"],
                "gold_answer": df["answer"],
                "output": [text for text, _ in outputs],
                "truncated": [trunc for _, trunc in outputs],
            })
            out_df = pd.concat([out_df, pd.DataFrame(graded)], axis=1)
            out_df.to_json(os.path.join(args.output_dir, f"generations_{name}.jsonl"), orient="records", lines=True)

            metrics[name] = summarize(out_df)
            metrics[name]["seconds"] = elapsed
            if shots_df is not None:
                metrics[name]["few_shot_ids"] = shots_df["id"].tolist()
            print(f"[{name}] accuracy {metrics[name]['accuracy']:.4f} | lenient {metrics[name]['accuracy_lenient']:.4f}"
                  f" | no \\boxed {metrics[name]['no_boxed_rate']:.3f} | truncated {metrics[name]['truncated_rate']:.3f}"
                  f" | {elapsed:.0f}s")

        result = {
            "settings": {
                "base_model": MODEL_NAME, "checkpoint": args.checkpoint, "init_model": init_model,
                "shots": args.shots, "limit": args.limit, "max_new_tokens": args.max_new_tokens,
                "backend": args.backend, "dtype": args.dtype, "decoding": "greedy", "stop": STOP_STRINGS,
            },
            "metrics": metrics,
        }
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(to_builtin(result), f, indent=2)
        print(f"\nSaved to {args.output_dir}")
    finally:
        if merged_dir is not None:
            shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
