"""Tiny end-to-end smoke test for the GRPO pipeline (needs a GPU).

Run from the repo root (so datasets/... paths resolve):
    python quick_test_grpo.py
    python quick_test_grpo.py --steps 4 --n_train 64 --n_val 16 --output_dir /tmp/grpo_smoke

It reuses the real code in train_grpo.py (get_model, load_and_process_datasets, get_trainer, rewards),
but with its own small test hyperparameters below, so config/argument errors show up here instead of in a long run.
"""
import argparse
import math
import time
import torch
import transformers
import trl
import peft
import vllm

import train_grpo
from train_grpo import (
    get_model,
    load_and_process_datasets,
    get_trainer,
    extract_answer,
    correctness_reward,
    format_reward,
    STOP_STRINGS,
)
from common import BaselineAndSummarizationTaskCallback
from config import GRPOTrainingConfig, LoraHParams

# Throwaway values for the smoke test only -- NOT suggestions for real runs
TEST_LORA = LoraHParams(r=16, lora_alpha=32, lora_dropout=0.0)
TEST_TRAINING = dict(
    num_train_epochs=1, learning_rate=1e-5, optim="adamw_torch_fused", weight_decay=0.0, max_grad_norm=1.0,
    lr_scheduler_type="constant", warmup_steps=0.0,
    per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=1,
    gradient_checkpointing=True, logging_steps=1,
    num_generations=4, max_completion_length=256, temperature=1.0, beta=0.0, loss_type="dapo", epsilon=0.2,
    num_iterations=1, scale_rewards="group", reward_weights=[1.0, 0.1],
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3, log_completions=False,
    save_strategy="no", eval_strategy="no", eval_on_start=False,
)

results = []

def section(title):
    print(f"\n===== {title} =====", flush=True)

def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_val", type=int, default=16)
    p.add_argument("--init_model", default=None, help="Optional SFT checkpoint, to smoke-test the SFT-start path")
    p.add_argument("--output_dir", default="/tmp/grpo_smoke")
    args = p.parse_args()
    test_config = GRPOTrainingConfig().model_copy(update=TEST_TRAINING)

    # 1. Environment
    section("Environment")
    print("torch", torch.__version__, "| transformers", transformers.__version__, "| trl", trl.__version__,
          "| peft", peft.__version__, "| vllm", vllm.__version__)
    check("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
        check("bf16 supported", torch.cuda.is_bf16_supported())

    # 2. Reward functions on hand-written completions (no model needed)
    section("Reward functions")
    answers = ["\\frac{3}{4}"] * 4
    completions = [" so $\\boxed{0.75}$.", " so $\\boxed{\\frac{4}{3}}$.", " so 3/4.",
                   " ok.\nQuestion: next\nSolution: $\\boxed{\\frac{3}{4}}$"]
    correct = correctness_reward(completions, answers, ["math"] * 4)
    fmt = format_reward(completions)
    print("correctness:", correct, "| format:", fmt)
    check("rewards on hand-written cases", correct == [1.0, 0, 0, 0] and fmt == [1.0, 1.0, 0, 0])

    # 3. Model + tokenizer
    section("Model and tokenizer")
    tokenizer, model = get_model(args.init_model, training_config=test_config, lora_config=TEST_LORA)
    print("pad:", tokenizer.pad_token, tokenizer.pad_token_id, "| eos:", tokenizer.eos_token, tokenizer.eos_token_id)
    dtypes = {str(p.dtype) for p in model.parameters()}
    check("model loaded in bf16", "torch.bfloat16" in dtypes, f"param dtypes: {dtypes}")

    # 4. Data (tiny slice)
    section("Data")
    train_ds, eval_ds = load_and_process_datasets(tokenizer)
    print("full sizes:", len(train_ds), len(eval_ds))
    train_ds = train_ds.shuffle(seed=0).select(range(args.n_train))
    eval_ds = eval_ds.shuffle(seed=0).select(range(args.n_val))
    print("slice sources:", {s: train_ds["source"].count(s) for s in set(train_ds["source"])})
    check("columns are prompt/answer/source", train_ds.column_names == ["prompt", "answer", "source"])

    # 5. Trainer from the real get_trainer, with the test config
    section("Trainer")
    t = time.time()
    trainer = get_trainer(tokenizer, model, train_ds, eval_ds, args.output_dir,
                          strategy="grpo_smoke", training_config=test_config)
    print(f"trainer built in {time.time() - t:.0f}s (includes starting vLLM)")
    trainer.remove_callback(BaselineAndSummarizationTaskCallback)  # no summarization here
    trainer.args.max_steps = args.steps
    check("GRPOConfig accepted every config field", True)
    check("vLLM stop string set", trainer.args.generation_kwargs == {"stop": STOP_STRINGS},
          f"generation_kwargs={trainer.args.generation_kwargs}")

    # 6. One group of sampled completions for one prompt, before training
    section("One group of completions (HF generate, before training)")
    ex = train_ds[0]
    enc = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=256,
                             num_return_sequences=test_config.num_generations, pad_token_id=tokenizer.pad_token_id)
    group = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    print("PROMPT:", ex["prompt"][:300], "\nANSWER:", ex["answer"])
    for i, c in enumerate(group):
        print(f"--- completion {i} | extracted={extract_answer(c)!r} | "
              f"correct={correctness_reward([c], [ex['answer']], [ex['source']])[0]}\n{c[:400]!r}")

    # 7. A few training steps
    section(f"Training {args.steps} steps")
    t = time.time()
    trainer.train()
    train_time = time.time() - t
    logs = [h for h in trainer.state.log_history if "loss" in h]
    keys = ["step", "loss", "reward", "reward_std", "frac_reward_zero_std", "rewards/correctness_reward/mean",
            "rewards/format_reward/mean", "accuracy/gsm8k", "accuracy/math",
            "completions/mean_length", "completions/clipped_ratio"]
    for h in logs:
        print({k: round(h[k], 4) if isinstance(h[k], float) else h[k] for k in keys if k in h})
    print(f"{train_time:.0f}s for {args.steps} steps ({train_time / max(args.steps, 1):.1f}s/step)")
    check("ran all steps", trainer.state.global_step == args.steps, f"global_step={trainer.state.global_step}")
    check("loss is finite", logs and all(math.isfinite(h["loss"]) for h in logs))
    check("some group had different rewards (learning signal)",
          logs and any(h.get("frac_reward_zero_std", 1.0) < 1.0 for h in logs))
    check("per-source accuracy logged", logs and any(k.startswith("accuracy/") for h in logs for k in h))

    # 8. Built-in GRPO eval (generates on the val slice and scores with the rewards)
    section("Eval")
    metrics = trainer.evaluate()
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    check("eval reports reward and accuracy", "eval_reward" in metrics and any(k.startswith("eval_accuracy/") for k in metrics))

    # 9. Memory
    if torch.cuda.is_available():
        section("Memory")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"peak allocated by torch: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB of {total:.0f} GB "
              f"(vLLM reserves ~{test_config.vllm_gpu_memory_utilization:.0%} separately)")

    section("Summary")
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("\nSMOKE TEST FINISHED:", "ALL PASSED" if all(ok for _, ok in results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
