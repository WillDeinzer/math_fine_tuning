import torch
from rouge_score import rouge_scorer
from bert_score import score
import pandas as pd
import os
import json

def create_prompt(article):
    prompt = f"Article: {article}\n\nSummarize the article above in a few sentences.\nSummary:"
    return prompt

def clean(text):
    for stop in ["\n\n", "Article:"]:
        text = text.split(stop)[0]
    return text.strip()

@torch.no_grad()
def generate_summaries(model, tokenizer, prompts, max_new_tokens=128, batch_size=4):
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    generations = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        generated_tokens = model.generate(**batch_tokens, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        generated_texts = tokenizer.batch_decode(generated_tokens[:, batch_tokens.input_ids.shape[1]:], skip_special_tokens=True)
        generations += [(t, clean(t)) for t in generated_texts]
    tokenizer.padding_side = prev_side
    return generations

def score_summaries(highlights, raw_generations, clean_generations, epoch):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(clean_generations, highlights)]
    average_rouge = sum(rouge_scores) / len(rouge_scores)

    torch.cuda.empty_cache()
    P, R, F1 = score(clean_generations, highlights, lang="en", verbose=False)
    average_bert_F1 = F1.mean().item()
    average_bert_precision = P.mean().item()
    average_bert_recall = R.mean().item()

    result = {
        "epoch": epoch,
        "rougeL": average_rouge,
        "bert_F1": average_bert_F1,
        "bert_precision": average_bert_precision,
        "bert_recall": average_bert_recall
    }
    result["mean_raw_words"] = sum(len(g.split()) for g in raw_generations) / len(raw_generations)
    result["mean_clean_words"] = sum(len(g.split()) for g in clean_generations) / len(clean_generations)
    result["empty_rate"] = sum(g == "" for g in clean_generations) / len(clean_generations)

    return result

def score_summarization_task(model, tokenizer, epoch, fine_tuning_strategy="sft_only"):
    df = pd.read_json("datasets/cnn_dm_eval.jsonl", lines=True)
    prompts = df["article"].apply(create_prompt).tolist()
    raw_and_clean = generate_summaries(model, tokenizer, prompts, max_new_tokens=128, batch_size=4)
    raw_generations = [r for r, _ in raw_and_clean]
    clean_generations = [c for _, c in raw_and_clean]
    out_dir = os.path.join("results", "summarization", fine_tuning_strategy, f"epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)
    gen_df = pd.DataFrame({
        "id": df["id"],
        "highlights": df["highlights"],
        "raw_generations": raw_generations,
        "clean_generations": clean_generations,
    })
    gen_df.to_json(os.path.join(out_dir, "generations.jsonl"), orient="records", lines=True)

    result = score_summaries(df["highlights"].tolist(), raw_generations, clean_generations, epoch)
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result

