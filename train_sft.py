from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import pandas as pd
import torch
from datasets import Dataset
import argparse
from common import build_prompt, drop_empty_answers, keep_short, BaselineAndSummarizationTaskCallback
from config import MODEL_NAME, TRAIN_PATH, VAL_PATH, SFTLoraConfig, SFTTrainingConfig

def get_model(training_config=SFTTrainingConfig(), lora_config=SFTLoraConfig()):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if training_config.bf16 else torch.float16,
        device_map="auto"
    )
    peft_config = LoraConfig(**lora_config.peft_kwargs())
    model = get_peft_model(model, peft_config)
    return tokenizer, model

def get_trainer(tokenizer, model, train_dataset, eval_dataset, output_dir="./sft_checkpoints", training_config=SFTTrainingConfig()):
    training_args = SFTConfig(output_dir=output_dir, **training_config.trainer_kwargs())

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[BaselineAndSummarizationTaskCallback(tokenizer)]
    )

    return trainer

def load_and_process_datasets(tokenizer):
    train_df = pd.read_json(TRAIN_PATH, lines=True, dtype={"answer": str})
    val_df = pd.read_json(VAL_PATH, lines=True, dtype={"answer": str})
    train_df = keep_short(drop_empty_answers(train_df), tokenizer)
    val_df = keep_short(drop_empty_answers(val_df), tokenizer)
    train_dataset = Dataset.from_dict({
        "prompt": train_df["question"].apply(build_prompt).tolist(),
        "completion": [" " + s + tokenizer.eos_token for s in train_df["solution"]],
    })
    eval_dataset = Dataset.from_dict({
        "prompt": val_df["question"].apply(build_prompt).tolist(),
        "completion": [" " + s + tokenizer.eos_token for s in val_df["solution"]],
    })
    return train_dataset, eval_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./sft_checkpoints")
    return p.parse_args()

def main():
    args = parse_args()
    tokenizer, model = get_model()
    train_dataset, eval_dataset = load_and_process_datasets(tokenizer)
    trainer = get_trainer(tokenizer, model, train_dataset, eval_dataset, args.output_dir)
    trainer.train()

if __name__ == "__main__":
    main()