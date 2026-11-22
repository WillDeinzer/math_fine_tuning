import os
import wandb
from transformers import TrainerCallback
from summarization_eval import score_summarization_task

def build_prompt(question):
    return f"Question: {question}\nSolution:"

def drop_empty_answers(df):
    keep = ~df["answer"].str.strip().isin(["", "None"])
    print(f"Dropped {len(df) - keep.sum()} / {len(df)} with an empty or 'None' answer")
    return df[keep]

def keep_short(df, tokenizer, max_len=1024):
    lens = [len(tokenizer(build_prompt(q) + " " + s).input_ids) + 1
            for q, s in zip(df["question"], df["solution"])]
    keep = [l <= max_len for l in lens]
    print(f"Dropped {len(df) - sum(keep)} / {len(df)} over {max_len} tokens")
    return df[keep]

class BaselineAndSummarizationTaskCallback(TrainerCallback):
    def __init__(self, tokenizer, fine_tuning_strategy="sft_only", adapter_dir=None):
        self.tokenizer = tokenizer
        self.fine_tuning_strategy = fine_tuning_strategy
        # After each eval the adapter is saved to <adapter_dir>/epoch_<N>; None skips saving
        self.adapter_dir = adapter_dir

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        epoch = 0 if state.epoch is None else round(state.epoch)
        print(f"\nRunning Summarization Task at Step {state.global_step} (Epoch {epoch})")

        model.eval()
        try:
            result = self.evaluate_summarization_task(model, self.tokenizer, epoch)

            if wandb.run is not None:
                wandb.log({
                    "summarization/rougeL": result["rougeL"],
                    "summarization/bert_F1": result["bert_F1"],
                    "summarization/bert_precision": result["bert_precision"],
                    "summarization/bert_recall": result["bert_recall"],
                    "epoch": epoch,
                }, step=state.global_step)

            if self.adapter_dir is not None:
                save_path = os.path.join(self.adapter_dir, f"epoch_{epoch}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"Saved adapter to {save_path}")
        finally:
            model.train()

        return control

    def evaluate_summarization_task(self, model, tokenizer, epoch):
        result = score_summarization_task(model, tokenizer, epoch, self.fine_tuning_strategy)

        print(
            f"\n[Summarization | {self.fine_tuning_strategy} | epoch {epoch}] "
            f"ROUGE-L {result['rougeL']:.4f} | "
            f"BERTScore F1 {result['bert_F1']:.4f} "
            f"(P {result['bert_precision']:.4f} / R {result['bert_recall']:.4f})"
        )
        return result
