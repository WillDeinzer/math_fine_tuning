from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from math_verify import parse, verify
import pandas as pd
import torch
from datasets import Dataset
import argparse
import os
import json
import re
from functools import lru_cache
from common import build_prompt, drop_empty_answers, keep_short, BaselineAndSummarizationTaskCallback
from config import MODEL_NAME, TRAIN_PATH, VAL_PATH, GRPOLoraConfig, GRPOTrainingConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./grpo_checkpoints")
    p.add_argument("--init_model", default=None, help="Path to an SFT checkpoint (LoRA adapter) to start from. Omit to start from the base model.")
    p.add_argument("--run_name", default=None, help="wandb run name (default: grpo_base or grpo_from_sft)")
    args = p.parse_args()

    if args.init_model is not None and not os.path.isdir(args.init_model):
        p.error(f"--init_model {args.init_model} is not a directory")

    args.strategy = "grpo_from_sft" if args.init_model else "grpo_base"
    args.run_name = args.run_name or args.strategy
    return args

def get_model(init_model=None, training_config=GRPOTrainingConfig(), lora_config=GRPOLoraConfig()):
    peft_config = LoraConfig(**lora_config.peft_kwargs())
    if init_model is not None and not os.path.exists(os.path.join(init_model, "adapter_config.json")):
        raise ValueError(f"{init_model} has no adapter_config.json; point --init_model at a checkpoint-N folder")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if training_config.bf16 else torch.float16,
        device_map="auto"
    )

    if init_model is not None:
        model = PeftModel.from_pretrained(model, init_model)
        model = model.merge_and_unload()
        print(f"Merged SFT adapter from {init_model} into the base model")

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return tokenizer, model

def to_grpo_dataset(df):
    return Dataset.from_dict({
        "prompt": df["question"].apply(build_prompt).tolist(),
        "answer": df["answer"].tolist(),
        "source": df["source"].tolist(),
    })

def load_and_process_datasets(tokenizer):
    train_df = pd.read_json(TRAIN_PATH, lines=True, dtype={"answer": str})
    val_df = pd.read_json(VAL_PATH, lines=True, dtype={"answer": str})
    train_df = keep_short(drop_empty_answers(train_df), tokenizer)
    val_df = keep_short(drop_empty_answers(val_df), tokenizer)
    train_dataset = to_grpo_dataset(train_df)
    eval_dataset = to_grpo_dataset(val_df)

    # useful for debugging
    example = train_dataset[0]
    print("=== SAMPLE GRPO EXAMPLE ===")
    print("PROMPT:", example["prompt"])
    print("ANSWER:", example["answer"], "| SOURCE:", example["source"])
    print("=== END SAMPLE ===")
    return train_dataset, eval_dataset

STOP_STRINGS = ["\nQuestion:"]

def last_boxed_content(text):
    start = text.rfind("\\boxed")
    if start == -1:
        return None
    i = start + len("\\boxed")

    if text[i:i + 1] == "{":
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j]
        return None

    bare = re.match(r"\s+([^\s$]+)", text[i:])
    return bare.group(1) if bare else None

def extract_answer(completion):
    for stop in STOP_STRINGS:
        completion = completion.split(stop, 1)[0]
    answer = last_boxed_content(completion)
    if answer is None or answer.strip() == "":
        return None
    return answer.strip()

@lru_cache(maxsize=None)
def parse_gold(answer):
    return parse(f"${answer}$")

def is_correct(completion, answer):
    pred = extract_answer(completion)
    if pred is None:
        return False
    try:
        return verify(parse_gold(answer), parse(f"${pred}$"))
    except Exception:
        return False

def correctness_reward(completions, answer, source, log_metric=None, **kwargs):
    correct = [is_correct(c, a) for c, a in zip(completions, answer)]
    if log_metric is not None:
        for is_right, src in zip(correct, source):
            log_metric(f"accuracy/{src}", float(is_right))
    return [1.0 if c else 0.0 for c in correct]

def format_reward(completions, **kwargs):
    return [1.0 if extract_answer(c) is not None else 0.0 for c in completions]

REWARD_FUNCS = [correctness_reward, format_reward]

def get_trainer(tokenizer, model, train_dataset, eval_dataset, output_dir="./grpo_checkpoints",
                strategy="grpo_base", training_config=GRPOTrainingConfig(), run_name=None):
    kwargs = training_config.trainer_kwargs()
    kwargs["run_name"] = run_name or strategy

    if kwargs["use_vllm"]:
        kwargs["generation_kwargs"] = {"stop": STOP_STRINGS}
    else:
        print(f"Note: use_vllm=False, so generation won't stop at {STOP_STRINGS}; "
              "completions run to max_completion_length (the reward still ignores anything after the stop string)")

    training_args = GRPOConfig(output_dir=output_dir, **kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[BaselineAndSummarizationTaskCallback(tokenizer, fine_tuning_strategy=strategy)]
    )

    return trainer

def save_run_info(args):
    os.makedirs(args.output_dir, exist_ok=True)
    info = {"base_model": MODEL_NAME, "init_model": args.init_model, "strategy": args.strategy, "run_name": args.run_name}
    with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

def main():
    args = parse_args()
    tokenizer, model = get_model(args.init_model)
    train_dataset, eval_dataset = load_and_process_datasets(tokenizer)
    trainer = get_trainer(tokenizer, model, train_dataset, eval_dataset, args.output_dir, args.strategy,
                          run_name=args.run_name)
    save_run_info(args)
    trainer.train()

if __name__ == "__main__":
    main()
