#!/usr/bin/env python3
from pathlib import Path
import argparse
import glob
import pandas as pd
from ranx import Qrels, Run, evaluate, compare

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # .../quantitative_eval
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "target" / "runs"

def _resolve_run(p: str) -> Path:
    """Resolve a single run file path by checking common roots."""
    cand = Path(p)
    if cand.exists():
        return cand.resolve()

    # Try relative to script dir, project root, and default runs dir
    for base in (HERE, ROOT, RUNS_DIR):
        maybe = (base / p).resolve()
        if maybe.exists():
            return maybe

    # Try just the filename inside runs dir
    fname = Path(p).name
    maybe = (RUNS_DIR / fname).resolve()
    if maybe.exists():
            return maybe

    raise FileNotFoundError(
        f"Run not found: {p}\nChecked:\n  {HERE}\n  {ROOT}\n  {RUNS_DIR}"
    )

def _expand_runs(args_runs: list[str]) -> list[Path]:
    """Expand globs and resolve loose paths into a de-duplicated list of Paths."""
    run_paths: list[Path] = []
    seen = set()

    for arg in args_runs:
        # First try glob expansion
        expanded = glob.glob(arg)
        if expanded:
            for e in expanded:
                p = Path(e).resolve()
                if p.exists() and p not in seen:
                    run_paths.append(p)
                    seen.add(p)
            continue

        # If no glob hits, resolve as a single file
        p = _resolve_run(arg)
        if p not in seen:
            run_paths.append(p)
            seen.add(p)

    if not run_paths:
        raise FileNotFoundError(f"No run files matched: {args_runs}")
    return run_paths

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate one or more TREC run files against qrels using ranx."
    )
    ap.add_argument(
        "--qrels",
        default=str(DATA_DIR / "qrels.txt"),
        help="Qrels file (default: DATA_DIR/qrels.txt)",
    )
    ap.add_argument(
        "runs",
        nargs="*",
        help="Run files or globs (default: tc.rag.dense.faiss.txt)",
    )
    args = ap.parse_args()

    # if no run paths given, fall back to a default run file
    if not args.runs:
        default_run = ROOT / "target" / "runs" / "tc.rag.dense.faiss.txt"
        if default_run.exists():
            args.runs = [str(default_run)]
        else:
            ap.error("No run file specified and default run file not found.")

    # Read qrels (qid \t subtopic \t docid \t rel)
    qrels_df = pd.read_csv(
        args.qrels,
        sep="\t",
        header=None,
        names=["q_id", "zero", "doc_id", "score"],
        encoding="cp1252",  # tolerant of Windows-1252 chars
    )
    qrels = Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")

    # Expand & resolve run files
    run_paths = _expand_runs(args.runs)

    # Load runs (trec format) and set names for your ranx version
    run_objs = []
    used = set()
    for p in run_paths:
        r = Run.from_file(str(p), kind="trec")
        # Ensure each Run has a unique .name for compare()
        base = p.name
        name = base
        i = 2
        while name in used:
            name = f"{base}#{i}"
            i += 1
        r.name = name
        used.add(name)
        run_objs.append(r)

    # Metrics — add more if you like
    metrics = ["ndcg@1", "ndcg@3", "ndcg@5"]

    # If a single run, also print the simple metric lines (handy for quick copy/paste)
    if len(run_objs) == 1:
        scores = evaluate(qrels=qrels, run=run_objs[0], metrics=metrics, make_comparable=True)
        print("All Topics")
        print("Run:", run_paths[0])
        for m in metrics:
            print(f"{m}: {scores[m]:.4f}")

    # Unified comparison report (works for 1 or many runs)
    report = compare(
        qrels=qrels,
        runs=run_objs,            # <- list of Run objects, each with .name set
        metrics=metrics,
        make_comparable=True,     # drops topics not in qrels; fills missing with empty
        rounding_digits=4,
    )

    print("All Topics")
    print(report)
    #print(report.to_latex())

if __name__ == "__main__":
    main()
