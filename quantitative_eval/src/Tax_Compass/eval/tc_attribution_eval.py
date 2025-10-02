#!/usr/bin/env python3
from pathlib import Path
import argparse, re, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # this is quantitative_eval\
DATA_DIR = ROOT / "data"
SUM_DIR = ROOT / "target" / "summaries"
CITE_RE = re.compile(r"\[(P\d+)\]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str((SUM_DIR / "tc_dense_eval_k3.csv").resolve()))
    ap.add_argument("--qrels",   default=str((DATA_DIR / "qrels.txt").resolve()))
    ap.add_argument("--out",     default=str((SUM_DIR / "tc_attrib_eval_k3.csv").resolve()))
    args = ap.parse_args()

    ans = pd.read_csv(args.answers)  # expects question_id, answer
    qrels = pd.read_csv(args.qrels, sep="\t", header=None, names=["qid","zero","docid","rel"])

    gold_map = qrels[qrels["rel"] > 0].groupby("qid")["docid"].apply(set).to_dict()

    rows = []
    for _, r in ans.iterrows():
        qid = r["question_id"]; a = r["answer"]
        cited = set(CITE_RE.findall(a))
        gold = gold_map.get(qid, set())

        tp = len(cited & gold)
        fp = len(cited - gold)
        fn = len(gold - cited)

        prec = tp / (tp+fp) if (tp+fp) else 0.0
        rec  = tp / (tp+fn) if (tp+fn) else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

        rows.append({
            "question_id": qid,
            "cited_ids": "|".join(sorted(cited)),
            "gold_ids":  "|".join(sorted(gold)),
            "attrib_precision": prec,
            "attrib_recall": rec,
            "attrib_f1": f1
        })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"[ok] wrote: {out_path}")

if __name__ == "__main__":
    main()
