#!/usr/bin/env python3
from pathlib import Path
import argparse, pandas as pd
from pyserini.search.faiss import FaissSearcher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # this is quantitative_eval\
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "target" / "indexes" / "tct_colbert-v2-hnp-msmarco-faiss-TaxCompass"
OUT_PREFIX = (ROOT / "target/chunking/FAISS_k").resolve()
ENCODER = "facebook/dpr-question_encoder-multiset-base"

def build(topics, collection, searcher, k):
    rows = []
    for qid, q in topics[["question_id", "question"]].values:
        hits = searcher.search(q, 10)
        top = hits[:k]
        ids, texts = [], []
        for h in top:
            ids.append(h.docid)
            txt = collection.loc[collection["passage_id"] == h.docid, "passage"].iloc[0]
            texts.append(txt)
        rows.append({"question_id": qid, "question": q, "topk_ids": ids, "topk_content": texts})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(INDEX_DIR))
    ap.add_argument("--topics", default=str(DATA_DIR / "topics.csv"))
    ap.add_argument("--collection", default=str(DATA_DIR / "collection.csv"))
    ap.add_argument("--out_prefix", default=str(OUT_PREFIX))
    ap.add_argument("--encoder", default=ENCODER)
    args = ap.parse_args()

    topics = pd.read_csv(args.topics, encoding="latin1")
    collection = pd.read_csv(args.collection, encoding="latin1")
    searcher = FaissSearcher(args.index, args.encoder)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for k in [1, 3, 5]:
        df = build(topics, collection, searcher, k)
        (out_prefix.parent).mkdir(parents=True, exist_ok=True)
        df.to_json(f"{out_prefix}{k}.jsonl", orient="records", lines=True)
        flat = df[["question_id", "question"]].copy()
        flat[f"top{k}_ids"] = df["topk_ids"].apply(lambda xs: "|".join(xs))
        flat[f"top{k}_content"] = df["topk_content"].apply(lambda vs: " || ".join(vs))
        flat.to_csv(f"{out_prefix}{k}.csv", index=False)
        print(f"[ok] wrote {out_prefix}{k}.jsonl and .csv")

if __name__ == "__main__":
    main()
