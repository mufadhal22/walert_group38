#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from pyserini.search.faiss import FaissSearcher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "target" / "indexes" / "tct_colbert-v2-hnp-msmarco-faiss-TaxCompass"
TOPICS = DATA_DIR / "topics.csv"
QRELS = DATA_DIR / "qrels.txt"
RUN_OUT = ROOT / "target" / "runs" / "tc.rag.dense.faiss.txt"
ENCODER = "facebook/dpr-question_encoder-multiset-base"

def load_qrels_qids(qrels_path: Path) -> set[str]:
    # qrels format: qid \t subtopic \t docid \t rel
    df = pd.read_csv(qrels_path, sep="\t", header=None,
                     names=["question_id", "subtopic", "passage_id", "relevance_judgment"],
                     encoding="cp1252")
    return set(df["question_id"].astype(str).unique())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(INDEX_DIR))
    ap.add_argument("--topics", default=str(TOPICS))
    ap.add_argument("--qrels", default=str(QRELS))
    ap.add_argument("--out", default=str(RUN_OUT))
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--restrict-to-qrels", action="store_true",
                    help="Only write run lines for topics that appear in qrels.txt")
    args = ap.parse_args()

    topics = pd.read_csv(args.topics, encoding="cp1252")
    topics["question_id"] = topics["question_id"].astype(str)

    allowed_qids = None
    if args.restrict_to_qrels:
        allowed_qids = load_qrels_qids(Path(args.qrels))
        topics = topics[topics["question_id"].isin(allowed_qids)].copy()

    searcher = FaissSearcher(args.index, args.encoder)
    runid = "tc.rag.dense.faiss"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="cp1252") as out:
        for qid, q in topics[["question_id", "question"]].itertuples(index=False):
            # if not restricting at load-time, you could still guard here:
            if allowed_qids is not None and qid not in allowed_qids:
                continue
            hits = searcher.search(q, args.depth)
            for rank, h in enumerate(hits, start=1):
                out.write(f"{qid} Q0 {h.docid} {rank} {h.score:.6f} {runid}\n")

    print(f"[ok] wrote run: {out_path} ({len(topics)} topics{' (judged-only)' if args.restrict_to_qrels else ''})")

if __name__ == "__main__":
    main()
