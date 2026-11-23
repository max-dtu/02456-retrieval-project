"""Print a table of all runs and their metrics.

Usage:
    python summarize_results.py                    # every batch under results/
    python summarize_results.py results/<batch>    # one batch
    python summarize_results.py --sort test_mrr    # sort by another column
"""
import argparse
import glob
import json
import os


def status_of(run_dir):
    """Best-effort job status from the LSF stdout file."""
    out = os.path.join(run_dir, "stdout.out")
    if not os.path.exists(out):  # older batches kept logs in <batch>/logs/<run>.out
        out = os.path.join(os.path.dirname(run_dir), "logs", os.path.basename(run_dir) + ".out")
    if not os.path.exists(out):
        return "queued/running"
    with open(out, errors="replace") as f:
        text = f.read()
    if "Successfully completed." in text:
        return "done"
    if "Exited with exit code" in text or "TERM_" in text:
        return "FAILED (see stderr.err)"
    return "running"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", default=["results"])
    parser.add_argument("--sort", default="val_mrr")
    args = parser.parse_args()

    run_dirs = set()
    for d in args.dirs:
        # a run dir contains .hydra/ or stdout.out; search up to two levels deep
        for pattern in ("", "*", "*/*"):
            for p in glob.glob(os.path.join(d, pattern)):
                if os.path.isdir(p) and (
                    os.path.exists(os.path.join(p, "stdout.out"))
                    or os.path.isdir(os.path.join(p, ".hydra"))
                ):
                    run_dirs.add(os.path.normpath(p))

    rows = []
    for rd in sorted(run_dirs):
        row = {"run": os.path.relpath(rd, "results") if rd.startswith("results") else rd}
        mpath = os.path.join(rd, "metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                metrics = json.load(f)
            for split, scores in metrics.items():
                for k, v in scores.items():
                    row[f"{split}_{k}"] = v
            row["status"] = "done"
        else:
            row["status"] = status_of(rd)
        rows.append(row)

    if not rows:
        print("No runs found.")
        return

    metric_cols = sorted({k for r in rows for k in r if k not in ("run", "status")},
                         key=lambda c: (["train", "val", "test"].index(c.split("_")[0])
                                        if c.split("_")[0] in ("train", "val", "test") else 9, c))
    rows.sort(key=lambda r: r.get(args.sort, float("-inf")), reverse=True)

    cols = ["run"] + metric_cols + ["status"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    widths = {c: max(len(c), *(len(fmt(r.get(c, "-"))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(fmt(r.get(c, "-")).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
