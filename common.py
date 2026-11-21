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
    def __init__(self, tokenizer, fine_tuning_strategy="sft_only"):
        self.tokenizer = tokenizer
        self.fine_tuning_strategy = fine_tuning_strategy

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        print(f"\n Running Summarization Task at Step {state.global_step} (Epoch {state.epoch})")
        model.eval()
        result = self.evaluate_summarization_task(model, self.tokenizer, state.epoch)

        if metrics is not None:
            metrics["eval_summ_rougeL"] = result["rougeL"]
            metrics["eval_summ_bert_F1"] = result["bert_F1"]

        model.train()

    def evaluate_summarization_task(self, model, tokenizer, epoch):
        epoch = 0 if epoch is None else round(epoch)
        result = score_summarization_task(model, tokenizer, epoch, self.fine_tuning_strategy)

        print(
            f"\n[Summarization | {self.fine_tuning_strategy} | epoch {epoch}] "
            f"ROUGE-L {result['rougeL']:.4f} | "
            f"BERTScore F1 {result['bert_F1']:.4f} "
            f"(P {result['bert_precision']:.4f} / R {result['bert_recall']:.4f})"
        )
        return result
