from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from summarization_eval import score_summarization_task
import pandas as pd
import torch
from datasets import Dataset
import argparse

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
LORA_DROPOUT = 0.1
LORA_ALPHA = 64
LORA_R = 64
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

class BaselineAndSummarizationTaskCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        print(f"\n Running Summarization Task at Step {state.global_step} (Epoch {state.epoch})")
        model.eval()
        result = self.evaluate_summarization_task(model, self.tokenizer, state.epoch)

        if metrics is not None:
            metrics["eval_summ_rougeL"] = result["rougeL"]
            metrics["eval_summ_bert_F1"] = result["bert_F1"]

        model.train()

    def evaluate_summarization_task(self, model, tokenizer, epoch, fine_tuning_strategy="sft_only"):
        epoch = 0 if epoch is None else round(epoch)
        result = score_summarization_task(model, tokenizer, epoch, fine_tuning_strategy)

        print(
            f"\n[Summarization | {fine_tuning_strategy} | epoch {epoch}] "
            f"ROUGE-L {result['rougeL']:.4f} | "
            f"BERTScore F1 {result['bert_F1']:.4f} "
            f"(P {result['bert_precision']:.4f} / R {result['bert_recall']:.4f})"
        )
        return result

def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    return tokenizer, model

def get_trainer(tokenizer, model, train_dataset, eval_dataset, output_dir="./sft_checkpoints"):
    num_train_epochs = 3
    optimizer = 'adamw_torch_fused'
    max_grad_norm = 1.0
    learning_rate = 1e-4
    weight_decay = 0.01 

    lr_scheduler_type = "cosine"
    warmup_ratio = 0.03
    fp16 = True
    bf16 = False
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    gradient_accumulation_steps = 8
    gradient_checkpointing = True
    save_strategy = 'epoch'
    eval_strategy='epoch'
    logging_steps = 25
    max_seq_length = 1024
    packing = False

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        eval_on_start=True,
        save_only_model=True,
        completion_only_loss=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optimizer,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_ratio,
        max_seq_length=max_seq_length,
        train_sampling_strategy="group_by_length",
        packing=packing,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[BaselineAndSummarizationTaskCallback(tokenizer)]
    )

    return trainer

def build_prompt(question):
    return f"Question: {question}\nSolution:"

def keep_short(df, tokenizer, max_len=1024):
    lens = [len(tokenizer(build_prompt(q) + " " + s).input_ids) + 1
            for q, s in zip(df["question"], df["solution"])]
    keep = [l <= max_len for l in lens]
    print(f"Dropped {len(df) - sum(keep)} / {len(df)} over {max_len} tokens")
    return df[keep]

def load_and_process_datasets(tokenizer):
    train_df = pd.read_json("datasets/train.jsonl", lines=True)
    val_df=pd.read_json("datasets/val.jsonl", lines=True)
    train_df = keep_short(train_df, tokenizer)
    val_df = keep_short(val_df, tokenizer)
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