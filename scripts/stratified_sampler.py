"""
Stratified sampler — selects a diverse cohort from a filtered persona set.

Stratification is configurable: pass dimension functions that map a row to a
bucket label. The sampler ensures minimum 1 per non-empty stratum, then fills
proportionally with within-stratum diversity on a secondary dimension.

Usage:
    uv run python scripts/stratified_sampler.py \
      --input data/filtered.json \
      --total 50 \
      --output data/cohort.json

    # Or with custom dimensions (as Python expressions)
    uv run python scripts/stratified_sampler.py \
      --input data/filtered.json \
      --total 50 \
      --dim-exprs '["age_bracket(r[\"age\"])", "r[\"marital_status\"]", "education_tier(r[\"education_level\"])"]'
"""

import json
import random
import argparse
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Built-in dimension functions ──────────────────────────────────────────

def age_bracket(age: int) -> str:
    if age <= 29: return "25-29"
    if age <= 34: return "30-34"
    if age <= 39: return "35-39"
    if age <= 49: return "40-49"
    return "50+"


def education_tier(edu: str) -> str:
    if edu in ("graduate",): return "graduate"
    if edu in ("bachelors",): return "bachelors"
    if edu in ("associates", "some_college"): return "some_college"
    return "no_degree"


def occupation_bucket(occ: str) -> str:
    occ = occ.lower()
    for kw in ("software", "computer", "data", "web", "engineer", "developer"):
        if kw in occ: return "tech"
    for kw in ("nurse", "doctor", "physician", "therapist", "health", "medical"):
        if kw in occ: return "healthcare"
    for kw in ("teacher", "professor", "instructor", "education"):
        if kw in occ: return "education"
    for kw in ("manager", "accountant", "financial", "analyst", "marketing", "sales"):
        if kw in occ: return "business"
    for kw in ("artist", "designer", "writer", "musician", "photographer"):
        if kw in occ: return "creative"
    for kw in ("cashier", "retail", "food", "customer", "secretary", "laborer"):
        if kw in occ: return "service"
    if occ in ("not in workforce", "no occupation", ""):
        return "not_working"
    return "other"


# ── Sampler ───────────────────────────────────────────────────────────────

def stratified_sample(profiles, dim_fns, total=50, diversity_fn=None, seed=42):
    """
    Stratified sample from profiles.

    Args:
        profiles: list of profile dicts
        dim_fns: list of callables, each takes a profile dict and returns a str label
        total: target sample size
        diversity_fn: optional callable for within-stratum diversity (takes profile, returns str)
        seed: random seed

    Returns:
        list of selected profile dicts
    """
    random.seed(seed)

    # Build strata
    strata = defaultdict(list)
    for p in profiles:
        key = tuple(fn(p) for fn in dim_fns)
        strata[key].append(p)

    print(f"Strata: {len(strata)} non-empty (from {len(profiles)} profiles)")

    # Allocate: min 1 per stratum, then proportional
    pop = sum(len(v) for v in strata.values())
    allocated = {k: 1 for k in strata}
    remaining = total - len(allocated)

    if remaining > 0:
        for key in sorted(strata, key=lambda k: len(strata[k]), reverse=True):
            extra = max(0, round(len(strata[key]) / pop * remaining))
            allocated[key] += extra

    # Cap total
    total_alloc = sum(allocated.values())
    if total_alloc > total:
        for key in sorted(allocated, key=lambda k: allocated[k], reverse=True):
            if total_alloc <= total:
                break
            trim = min(allocated[key] - 1, total_alloc - total)
            allocated[key] -= trim
            total_alloc -= trim

    # Sample with within-stratum diversity
    selected = []
    for key, n in allocated.items():
        members = strata[key]
        if n >= len(members):
            selected.extend(members)
        elif diversity_fn is None:
            selected.extend(random.sample(members, n))
        else:
            # Round-robin across diversity buckets
            by_bucket = defaultdict(list)
            for p in members:
                by_bucket[diversity_fn(p)].append(p)
            chosen = []
            buckets = list(by_bucket.keys())
            random.shuffle(buckets)
            bi = 0
            while len(chosen) < n and any(by_bucket.values()):
                b = buckets[bi % len(buckets)]
                if by_bucket[b]:
                    chosen.append(by_bucket[b].pop(random.randrange(len(by_bucket[b]))))
                bi += 1
                if bi > n * len(buckets):
                    break
            selected.extend(chosen)

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/filtered.json")
    parser.add_argument("--total", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/cohort.json")
    args = parser.parse_args()

    with open(args.input) as f:
        profiles = json.load(f)
    print(f"Loaded {len(profiles)} profiles from {args.input}")

    # Default dimensions: age, marital status, education
    dim_fns = [
        lambda p: age_bracket(p.get("age", 30)),
        lambda p: p.get("marital_status", "unknown"),
        lambda p: education_tier(p.get("education_level", "")),
    ]
    diversity_fn = lambda p: occupation_bucket(p.get("occupation", ""))

    selected = stratified_sample(profiles, dim_fns, total=args.total,
                                  diversity_fn=diversity_fn, seed=args.seed)

    # Re-assign user_ids
    for i, p in enumerate(selected):
        p["user_id"] = i

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\nSaved {len(selected)} to {args.output}")
    for dim_name, fn in [("Age", lambda p: age_bracket(p.get("age", 30))),
                          ("Marital", lambda p: p.get("marital_status", "?")),
                          ("Education", lambda p: education_tier(p.get("education_level", ""))),
                          ("Occupation", lambda p: occupation_bucket(p.get("occupation", "")))]:
        dist = Counter(fn(p) for p in selected)
        print(f"  {dim_name}: {dict(sorted(dist.items()))}")
    print(f"  Cities: {len(set(p.get('city','') for p in selected))} unique")


if __name__ == "__main__":
    main()
