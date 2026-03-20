"""
Cross-run comparison — track how changes to θ affect scores over time.

Usage:
    uv run python scripts/compare.py
    uv run python scripts/compare.py --runs baseline v2_with_freetier
"""

import json
import argparse
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_run(tag):
    d = RESULTS_DIR / tag
    with open(d / "raw_results.json") as f:
        results = json.load(f)
    with open(d / "meta.json") as f:
        meta = json.load(f)
    return meta, results


def summarize(results):
    valid = [r for r in results if "score" in r]
    if not valid:
        return {}
    scores = [r["score"] for r in valid]
    actions = [r["action"] for r in valid]
    n = len(valid)
    return {
        "n": n,
        "avg": round(sum(scores) / n, 1),
        "positive": actions.count("positive"),
        "neutral": actions.count("neutral"),
        "negative": actions.count("negative"),
        "pos_pct": round(100 * actions.count("positive") / n),
        "attractions": Counter(a for r in valid for a in r.get("attractions", [])).most_common(5),
        "concerns": Counter(c for r in valid for c in r.get("concerns", [])).most_common(5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    args = parser.parse_args()

    if args.runs:
        tags = args.runs
    else:
        tags = sorted(d.name for d in RESULTS_DIR.iterdir()
                      if d.is_dir() and (d / "meta.json").exists())

    if not tags:
        print("No runs found.")
        return

    print(f"{'='*75}")
    print(f"COMPARISON — {len(tags)} RUNS")
    print(f"{'='*75}\n")

    summaries = []
    for tag in tags:
        meta, results = load_run(tag)
        s = summarize(results)
        s["tag"] = tag
        s["entity"] = Path(meta.get("entity", "?")).name
        s["date"] = meta.get("timestamp", "?")[:10]
        summaries.append(s)

    print(f"{'Tag':<28} {'Date':<12} {'Entity':<22} {'Avg':>5} {'✅':>5} {'🤔':>5} {'❌':>5}")
    print("-" * 85)
    for s in summaries:
        print(f"{s['tag']:<28} {s['date']:<12} {s['entity']:<22} "
              f"{s['avg']:>5.1f} {s['positive']:>4}  {s['neutral']:>4}  {s['negative']:>4}")

    if len(summaries) >= 2:
        prev, curr = summaries[-2], summaries[-1]
        delta = curr["avg"] - prev["avg"]
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"\nDelta ({prev['tag']} → {curr['tag']}): {arrow} {delta:+.1f}")

        prev_a = set(a for a, _ in prev.get("attractions", []))
        curr_a = set(a for a, _ in curr.get("attractions", []))
        if curr_a - prev_a:
            print(f"  New attractions: {curr_a - prev_a}")
        if prev_a - curr_a:
            print(f"  Lost attractions: {prev_a - curr_a}")

        prev_c = set(c for c, _ in prev.get("concerns", []))
        curr_c = set(c for c, _ in curr.get("concerns", []))
        if curr_c - prev_c:
            print(f"  New concerns: {curr_c - prev_c}")
        if prev_c - curr_c:
            print(f"  Resolved concerns: {prev_c - curr_c}")


if __name__ == "__main__":
    main()
