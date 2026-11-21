"""Tiny end-to-end smoke test for the SFT pipeline.

Run from the repo root (so datasets/... paths resolve):
    python smoke_test.py
    python smoke_test.py --steps 10 --n_train 64 --n_val 16 --output_dir /tmp/smoke

It reuses the real code in train_sft.py (get_model, load_and_process_datasets,
get_trainer), so config/argument errors show up here instead of in a long run.
"""
import argparse
import torch
import transformers
import trl
import peft

import train_sft
from train_sft import (
    get_model,
    load_and_process_datasets,
    get_trainer,
)
from common import build_prompt, BaselineAndSummarizationTaskCallback


def section(title):
    print(f"\n===== {title} =====")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_val", type=int, default=16)
    p.add_argument("--output_dir", default="/tmp/smoke")
    args = p.parse_args()

    # 1. Environment
    section("Environment")
    print("torch", torch.__version__, "| transformers", transformers.__version__,
          "| trl", trl.__version__, "| peft", peft.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0),
              "| capability:", torch.cuda.get_device_capability(0),
              "| bf16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("WARNING: no CUDA device found")

    # 2. Model + tokenizer
    section("Model and tokenizer")
    tokenizer, model = get_model()
    print("pad:", tokenizer.pad_token, tokenizer.pad_token_id,
          "| eos:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("generation_config eos_token_id:", model.generation_config.eos_token_id)
    model.print_trainable_parameters()
    dtypes = {str(p.dtype) for p in model.parameters() if p.requires_grad}
    print("trainable param dtypes:", dtypes)

    # 3. Data (tiny slice)
    section("Data")
    train_ds, eval_ds = load_and_process_datasets(tokenizer)
    print("full sizes:", len(train_ds), len(eval_ds))
    train_ds = train_ds.shuffle(seed=0).select(range(args.n_train))
    eval_ds = eval_ds.shuffle(seed=0).select(range(args.n_val))
    print("example prompt:\n", train_ds[0]["prompt"][:300])
    print("example completion tail:\n", repr(train_ds[0]["completion"][-120:]))

    # 4. Trainer built from the real config, then shrunk for a smoke run
    section("Trainer")
    trainer = get_trainer(tokenizer, model, train_ds, eval_ds)
    trainer.remove_callback(BaselineAndSummarizationTaskCallback)  # no summarization here
    trainer.args.output_dir = args.output_dir
    trainer.args.max_steps = args.steps
    trainer.args.logging_steps = 1
    trainer.args.save_strategy = "no"
    trainer.args.eval_strategy = "no"
    trainer.args.eval_on_start = False

    # 5. Masking / EOS check (the most important one)
    section("Label masking check")
    batch = next(iter(trainer.get_train_dataloader()))
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    trained = ids[labels != -100]
    print("tokens in sequence:", len(ids), "| tokens with loss:", len(trained))
    print("trained text:\n", repr(tokenizer.decode(trained))[:400], "...")
    print("last 3 trained tokens:", tokenizer.convert_ids_to_tokens(trained[-3:].tolist()))
    n_eos = int((trained == tokenizer.eos_token_id).sum())
    print(f"EOS tokens in loss: {n_eos}  (want exactly 1)")
    prompt_leaked = "Question:" in tokenizer.decode(trained)
    print("prompt leaked into loss:", prompt_leaked, "(want False)")

    # 6. A few training steps
    section(f"Training {args.steps} steps")
    trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print("train losses:", [round(x, 3) for x in losses])

    # 7. Validation loss
    section("Validation loss")
    print(trainer.evaluate())

    # 8. Tiny generation
    section("Generation")
    model.eval()
    prompt = build_prompt("What is 12 plus 30?")
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=64, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    print(repr(tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)))

    # 9. Memory
    if torch.cuda.is_available():
        section("Memory")
        print(f"peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\nSMOKE TEST FINISHED")


if __name__ == "__main__":
    main()