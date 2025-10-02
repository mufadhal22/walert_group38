#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paths ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                    # .../quantitative_eval
DATA_DIR = ROOT / "data"
SUM_DIR = ROOT / "target" / "summaries"

# ---------- Helpers ----------
INSUF_MSG = "I don't have enough information to answer that."

def _ensure_str_id(df, col="question_id"):
    if col in df.columns:
        df[col] = df[col].astype(str)
    return df

def _safe_read_csv(path: Path, **kw):
    # tolerant encoding for Windows-created CSVs
    return pd.read_csv(path, encoding="cp1252", **kw)

def _load_topics(topics_path: Path) -> pd.DataFrame:
    topics = _safe_read_csv(topics_path)
    topics = _ensure_str_id(topics, "question_id")
    if "question" not in topics.columns:
        topics["question"] = ""
    return topics[["question_id", "question"]].drop_duplicates()

def _save_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def _hist_plot(series, title, out_path: Path, bins=20):
    plt.figure()
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _bar_plot(df, xcol, ycol, title, out_path: Path, rotate_xticks=True):
    plt.figure()
    plt.bar(df[xcol].astype(str), df[ycol])
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    if rotate_xticks:
        plt.xticks(rotation=45, ha="right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _scatter_plot(x, y, title, out_path: Path, xlabel="x", ylabel="y"):
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- Retrieval coverage from chunkset + qrels ----------
def _load_qrels(path: Path) -> dict:
    # qrels format: qid \t subtopic \t docid \t rel
    df = _safe_read_csv(path, sep="\t", header=None,
                        names=["question_id","subtopic","doc_id","rel"])
    df["question_id"] = df["question_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    grp = df.groupby("question_id")["doc_id"].apply(set)
    return grp.to_dict()

def _load_chunkset(path: Path):
    # returns list of dicts: {"question_id":..., "topk_ids":[...], "question":... (optional)}
    rows = []
    with open(path, "r", encoding="cp1252") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["question_id"] = str(row["question_id"])
            rows.append(row)
    return rows

def _retrieval_coverage_stats(chunk_rows, qrels_dict):
    records = []
    for row in chunk_rows:
        qid = row["question_id"]
        ids = [str(x) for x in row.get("topk_ids", [])]
        gold = qrels_dict.get(qid, set())
        rel_hits = [1 if i in gold else 0 for i in ids]
        rel_count = sum(rel_hits)
        # first relevant rank (1-based), or None
        first_rank = None
        for idx, hit in enumerate(rel_hits, start=1):
            if hit == 1:
                first_rank = idx
                break
        records.append({
            "question_id": qid,
            "k": len(ids),
            "relevant_in_topk": rel_count,
            "first_rel_rank": first_rank,
        })
    cov_df = pd.DataFrame(records)
    # hit-rate@k
    hit_rate = float((cov_df["relevant_in_topk"] > 0).mean()) if len(cov_df) else 0.0
    # distribution of #relevant in top-k
    dist = cov_df["relevant_in_topk"].value_counts().sort_index()
    # MRR@k (only count questions with a first relevant)
    mrr = 0.0
    if len(cov_df):
        rr = cov_df["first_rel_rank"].dropna().apply(lambda r: 1.0/float(r))
        mrr = float(rr.mean()) if len(rr) else 0.0
    return cov_df, hit_rate, dist, mrr

def main():
    ap = argparse.ArgumentParser(
        description="Summarize & visualize ROUGE/BERT + Attribution results, plus retrieval coverage and optional NDCG."
    )
    ap.add_argument("--k", type=int, default=3, help="K value used (default: 3)")
    ap.add_argument("--gen", default=None,
                    help="Path to gen eval CSV (tc_dense_eval_k{K}.csv).")
    ap.add_argument("--attrib", default=None,
                    help="Path to attribution CSV (tc_attrib_eval_k{K}.csv).")
    ap.add_argument("--topics", default=str(DATA_DIR / "topics.csv"),
                    help="Path to topics.csv for question text.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: target/summaries/analysis_k{K}/)")
    # New: retrieval coverage
    ap.add_argument("--chunkset", default=None,
                    help="JSONL with top-k retrieval per query (default: target/chunking/FAISS_k{K}.jsonl)")
    ap.add_argument("--qrels", default=str(DATA_DIR / "qrels.txt"),
                    help="Qrels file used for retrieval relevance.")
    # Optional: NDCG from a run file (if provided)
    ap.add_argument("--run", default=None,
                    help="Optional TREC run to compute NDCG in this script (requires ranx).")
    args = ap.parse_args()

    # Resolve inputs
    gen_path = Path(args.gen) if args.gen else (SUM_DIR / f"tc_dense_eval_k{args.k}.csv")
    attrib_path = Path(args.attrib) if args.attrib else (SUM_DIR / f"tc_attrib_eval_k{args.k}.csv")
    topics_path = Path(args.topics)
    outdir = Path(args.outdir) if args.outdir else (SUM_DIR / f"analysis_k{args.k}")
    chunkset_path = Path(args.chunkset) if args.chunkset else (ROOT / f"target/chunking/FAISS_k{args.k}.jsonl")
    qrels_path = Path(args.qrels)

    # Load data
    gen = _safe_read_csv(gen_path)
    gen = _ensure_str_id(gen, "question_id")
    attrib = _safe_read_csv(attrib_path)
    attrib = _ensure_str_id(attrib, "question_id")
    topics = _load_topics(topics_path)

    # Merge for rich tables
    gen_merged = gen.merge(topics, on="question_id", how="left")
    attrib_merged = attrib.merge(topics, on="question_id", how="left")

    # ----- Overall averages -----
    rouge_cols = ["rouge_1_f1", "rouge_2_f1", "rouge_l_f1"]
    bert_cols = ["bert_p", "bert_r", "bert_f1"]
    attrib_cols = ["attrib_precision", "attrib_recall", "attrib_f1"]

    overall = {}
    for c in rouge_cols + bert_cols:
        if c in gen_merged.columns:
            overall[c] = float(gen_merged[c].mean())
    for c in attrib_cols:
        if c in attrib_merged.columns:
            overall[c] = float(attrib_merged[c].mean())

    _save_csv(pd.DataFrame([overall]), outdir / "overall_averages.csv")

    # Console summary
    print("=== Overall Averages ===")
    for k_, v_ in overall.items():
        print(f"{k_:>16}: {v_:.4f}")

    # ----- Retrieval coverage / hit-rate from chunkset + qrels -----
    if chunkset_path.exists() and qrels_path.exists():
        qrels_map = _load_qrels(qrels_path)
        chunk_rows = _load_chunkset(chunkset_path)
        cov_df, hit_rate, dist, mrr = _retrieval_coverage_stats(chunk_rows, qrels_map)
        _save_csv(cov_df, outdir / "retrieval_coverage.csv")
        dist_df = dist.reset_index()
        dist_df.columns = ["relevant_in_topk", "count"]
        _save_csv(dist_df, outdir / "retrieval_relcount_distribution.csv")

        total = len(cov_df)
        zero_rel = int((cov_df["relevant_in_topk"] == 0).sum())
        one_or_more = total - zero_rel
        print("\n=== Retrieval Coverage (from chunkset & qrels) ===")
        print(f"Total topics: {total}")
        print(f"Hit@{args.k} (≥1 relevant in top-{args.k}): {one_or_more} ({(one_or_more/total*100 if total else 0):.1f}%)")
        print(f"Zero relevant in top-{args.k}: {zero_rel} ({(zero_rel/total*100 if total else 0):.1f}%)")
        print("Distribution of #relevant in top-k:")
        for rcount, cnt in dist.items():
            print(f"  {rcount}: {cnt}")
        print(f"MRR@{args.k}: {mrr:.4f}")
    else:
        print("\n[info] Skipping retrieval coverage: chunkset or qrels not found.")

    # ----- Optional NDCG from run (recompute here) -----
    if args.run:
        try:
            from ranx import Qrels, Run, evaluate
            qrels_df = _safe_read_csv(qrels_path, sep="\t", header=None,
                                      names=["q_id","zero","doc_id","score"])
            qrels = Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")
            run = Run.from_file(str(Path(args.run)), kind="trec")
            metrics = ["ndcg@1","ndcg@3","ndcg@5","ndcg@7","ndcg@9"]
            scores = evaluate(qrels=qrels, run=run, metrics=metrics, make_comparable=True)
            print("\n=== NDCG (from run) ===")
            for m in metrics:
                print(f"{m}: {scores[m]:.4f}")
        except Exception as e:
            print(f"\n[warn] Could not compute NDCG from run: {e}")

    # ----- Top/Bottom N tables -----
    N = 10
    def top_bottom(df, score_col, extra_cols):
        df2 = df.copy()
        df2 = df2[["question_id", "question", score_col] + extra_cols].copy()
        df2 = df2.sort_values(score_col, ascending=False)
        top = df2.head(N)
        bottom = df2.tail(N).sort_values(score_col, ascending=True)
        return top, bottom

    # ROUGE-L F1
    plots_dir = outdir / "plots"
    if "rouge_l_f1" in gen_merged.columns:
        rtop, rbot = top_bottom(gen_merged, "rouge_l_f1", [])
        _save_csv(rtop, outdir / "top_rougeL.csv")
        _save_csv(rbot, outdir / "bottom_rougeL.csv")
        _hist_plot(gen_merged["rouge_l_f1"], "ROUGE-L F1 distribution", plots_dir / "hist_rougeL.png")
        _bar_plot(rtop, "question_id", "rouge_l_f1", f"Top {N} ROUGE-L F1", plots_dir / "top_rougeL.png")
        _bar_plot(rbot, "question_id", "rouge_l_f1", f"Bottom {N} ROUGE-L F1", plots_dir / "bottom_rougeL.png")

    # BERT F1
    if "bert_f1" in gen_merged.columns:
        btop, bbot = top_bottom(gen_merged, "bert_f1", [])
        _save_csv(btop, outdir / "top_bertF1.csv")
        _save_csv(bbot, outdir / "bottom_bertF1.csv")
        _hist_plot(gen_merged["bert_f1"], "BERT F1 distribution", plots_dir / "hist_bertF1.png")
        _bar_plot(btop, "question_id", "bert_f1", f"Top {N} BERT F1", plots_dir / "top_bertF1.png")
        _bar_plot(bbot, "question_id", "bert_f1", f"Bottom {N} BERT F1", plots_dir / "bottom_bertF1.png")

    # Attribution F1 with cited/gold IDs
    extra_attrib_cols = [c for c in ["cited_ids","gold_ids"] if c in attrib_merged.columns]
    if "attrib_f1" in attrib_merged.columns:
        atop, abot = top_bottom(attrib_merged, "attrib_f1", extra_attrib_cols)
        _save_csv(atop, outdir / "top_attribF1.csv")
        _save_csv(abot, outdir / "bottom_attribF1.csv")
        _hist_plot(attrib_merged["attrib_f1"], "Attribution F1 distribution", plots_dir / "hist_attribF1.png")
        _bar_plot(atop, "question_id", "attrib_f1", f"Top {N} Attribution F1", plots_dir / "top_attribF1.png")
        _bar_plot(abot, "question_id", "attrib_f1", f"Bottom {N} Attribution F1", plots_dir / "bottom_attribF1.png")

    # Insufficient answers diagnostics
    insuf = (gen_merged["answer"].fillna("").str.strip() == INSUF_MSG)
    insuf_df = gen_merged.loc[insuf, ["question_id","question"]].copy()
    insuf_df["insufficient"] = 1
    _save_csv(insuf_df, outdir / "insufficient_answers.csv")
    print("\n=== Insufficient Information Answers ===")
    print(f"Count: {int(insuf.sum())}")

    # Citations behavior and correlation with attribution F1
    # Count [Pxx] occurrences per answer
    cite_counts = gen_merged["answer"].fillna("").str.count(r"\[P\d+\]")
    gen_merged["cite_count"] = cite_counts
    _hist_plot(cite_counts, "Citations per answer", plots_dir / "hist_citations_per_answer.png")

    if "attrib_f1" in attrib_merged.columns:
        corr_df = gen_merged.merge(attrib_merged[["question_id","attrib_f1"]], on="question_id", how="left")
        corr = corr_df["cite_count"].corr(corr_df["attrib_f1"], method="pearson")
        _save_csv(pd.DataFrame([{"pearson_citecount_vs_attribF1": float(corr)}]), outdir / "correlation_cites_attribF1.csv")
        _scatter_plot(corr_df["cite_count"], corr_df["attrib_f1"],
                      "Citations per answer vs Attribution F1",
                      plots_dir / "scatter_cites_vs_attribF1.png",
                      xlabel="# citations in answer", ylabel="Attribution F1")
        print("\n=== Citation Behavior ===")
        print(f"Mean citations per answer: {cite_counts.mean():.2f}")
        print(f"Pearson corr(citations, attribution F1): {corr:.3f}")

    # Error slices (bottom-N attribution with context)
    if "attrib_f1" in attrib_merged.columns:
        merged_all = attrib_merged.merge(
            gen_merged[["question_id","answer","rouge_l_f1","bert_f1","cite_count"]],
            on="question_id", how="left"
        )
        slice_cols = ["question_id","question","attrib_f1","cited_ids","gold_ids","answer","rouge_l_f1","bert_f1","cite_count"]
        error_df = merged_all.sort_values("attrib_f1", ascending=True)[slice_cols].head(N)
        _save_csv(error_df, outdir / "error_slices_bottom_attribF1.csv")

    print(f"\n[ok] Wrote summaries & plots to: {outdir}")

if __name__ == "__main__":
    main()
