#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re

import pandas as pd
import torch
from langchain_ollama import ChatOllama
from rouge import Rouge
from bert_score import score as bertscore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                    # .../quantitative_eval
DATA_DIR = ROOT / "data"
SUM_DIR = ROOT / "target" / "summaries"   # <-- updated location
MODEL = "llama3.2:latest"

PROMPT_TMPL = """You are Tax Compass. Answer ONLY with facts from the provided documents and CITE passage IDs in square brackets (e.g., [P01]).
If the documents are insufficient, reply exactly: "I don't have enough information to answer that."

Question: {question}

{ctx}

Answer (with citations):"""

# --- cleaning just for metrics (keep citations in saved CSV) ---
CITE_BRACKETS_RE = re.compile(r"\s*\[(?:P\d+)\]\s*")
DOC_LABEL_RE = re.compile(r"\bDocument\s+\d+\s*(?=\[P\d+\])?")

def strip_for_metrics(text: str) -> str:
    if not text:
        return ""
    t = DOC_LABEL_RE.sub("", text)     # drop "Document N" labels (optional)
    t = CITE_BRACKETS_RE.sub(" ", t)   # remove [Pxx] citations
    return " ".join(t.split())

def make_ctx(ids, texts):
    # LLM can still see "Document N", but we'll strip it for metric scoring only
    return "\n".join([f"Document {i} [{pid}]: {txt}"
                      for i, (pid, txt) in enumerate(zip(ids, texts), 1)])

def compute_rouge(hyp: str, ref: str):
    hyp = (hyp or "").strip()
    ref = (ref or "").strip()
    if not hyp or not ref:
        return 0.0, 0.0, 0.0
    r = Rouge().get_scores(hyp, ref, avg=True)
    return r["rouge-1"]["f"], r["rouge-2"]["f"], r["rouge-l"]["f"]

def compute_bert(hyp: str, ref: str):
    hyp = (hyp or "").strip()
    ref = (ref or "").strip()
    if not hyp or not ref:
        return 0.0, 0.0, 0.0
    P, R, F = bertscore(
        [hyp], [ref],
        lang="en",
        model_type="bert-base-uncased",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return P.item(), R.item(), F.item()

def load_golds(path: Path) -> pd.Series:
    """Return Series mapping question_id -> single summary (first occurrence)."""
    df = pd.read_csv(path, encoding="latin1").dropna(subset=["question_id", "summary"]).copy()
    df["question_id"] = df["question_id"].astype(str)
    ser = df.groupby("question_id", as_index=True)["summary"].first().astype(str)
    return ser

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", choices=["1", "3", "5"], default="3")
    ap.add_argument("--chunkset", default=None,
                    help="JSONL with top-k passages; default uses target/chunking/FAISS_k{K}.jsonl")
    ap.add_argument("--golds", default=str((DATA_DIR / "gold_summaries.csv").resolve()))
    ap.add_argument("--out", default=None,
                    help="CSV output path; default uses target/summaries/tc_dense_eval_k{K}.csv")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    # infer defaults that depend on --k
    k = int(args.k)
    chunkset_path = Path(args.chunkset) if args.chunkset else (ROOT / f"target/chunking/FAISS_k{k}.jsonl")
    out_path = Path(args.out) if args.out else (SUM_DIR / f"tc_dense_eval_k{k}.csv")

    SUM_DIR.mkdir(parents=True, exist_ok=True)

    llm = ChatOllama(model=args.model, temperature=0.0)
    golds = load_golds(Path(args.golds))  # Series: qid -> summary

    rows = []
    with open(chunkset_path, "r", encoding="latin1") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["question_id"])
            question = str(row["question"])
            ids = row["topk_ids"]
            texts = row["topk_content"]

            ctx = make_ctx(ids, texts)
            prompt = PROMPT_TMPL.format(question=question, ctx=ctx)
            ans = (llm.invoke(prompt).content or "").strip()

            # metrics on a cleaned copy (citations removed)
            ans_for_metrics = strip_for_metrics(ans)
            ref = golds.get(qid, "")
            r1, r2, rl = compute_rouge(ans_for_metrics, ref)
            bp, br, bf1 = compute_bert(ans_for_metrics, ref)

            rows.append({
                "question_id": qid,
                "k": k,
                "answer": ans,  # keep citations for attribution eval
                "rouge_1_f1": r1, "rouge_2_f1": r2, "rouge_l_f1": rl,
                "bert_p": bp, "bert_r": br, "bert_f1": bf1,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"[ok] wrote: {out_path}")

if __name__ == "__main__":
    main()
