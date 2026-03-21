"""
Stratified sampler — selects a diverse cohort from a filtered persona set.

Stratification is configurable: pass dimension functions that map a row to a
bucket label. The sampler ensures minimum 1 per non-empty stratum, then fills
proportionally with within-stratum diversity on a secondary dimension.

When --entity is provided, occupation bucketing is done via LLM: the full set
of unique occupation values is sent to the model along with the entity
description, so the grouping reflects how different professions would evaluate
that specific entity. The mapping is cached per entity.

Usage:
    uv run python scripts/stratified_sampler.py \
      --input data/filtered.json \
      --entity entities/my_product.md \
      --total 50 \
      --output data/cohort.json

    # Without entity (uses raw occupation values, no bucketing)
    uv run python scripts/stratified_sampler.py \
      --input data/filtered.json \
      --total 50 \
      --output data/cohort.json
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


def make_occupation_fn(entity_path=None, profiles=None):
    """
    Build an occupation bucketing function.

    With --entity: uses LLM to create a target-aware mapping from the full
    set of unique occupation values. Cached per entity content.

    Without --entity: passes through the raw occupation value.
    """
    if entity_path is None:
        return lambda p: p.get("occupation", "unknown") or "unknown"

    entity_text = Path(entity_path).read_text()

    # Collect unique occupation values from the profiles being sampled
    unique_occs = set()
    for p in (profiles or []):
        occ = p.get("occupation", "")
        if occ:
            unique_occs.add(occ)

    if not unique_occs:
        return lambda p: p.get("occupation", "unknown") or "unknown"

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_category_map import load_or_build_map
    mapping = load_or_build_map("occupation", entity_text, list(unique_occs))

    def lookup(p):
        occ = p.get("occupation", "")
        return mapping.get(occ, mapping.get(occ.lower(), "other"))

    return lookup


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
    parser.add_argument("--entity", default=None,
                        help="Path to entity document (enables LLM-based occupation bucketing)")
    parser.add_argument("--total", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/cohort.json")
    args = parser.parse_args()

    with open(args.input) as f:
        profiles = json.load(f)
    print(f"Loaded {len(profiles)} profiles from {args.input}")

    # Build occupation function — LLM-based if entity provided, raw passthrough otherwise
    occupation_fn = make_occupation_fn(args.entity, profiles)

    # Default dimensions: age, marital status, education (raw values)
    dim_fns = [
        lambda p: age_bracket(p.get("age", 30)),
        lambda p: p.get("marital_status", "unknown"),
        lambda p: p.get("education_level", "") or "unknown",
    ]
    diversity_fn = occupation_fn

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
                          ("Education", lambda p: p.get("education_level", "") or "unknown"),
                          ("Occupation", occupation_fn)]:
        dist = Counter(fn(p) for p in selected)
        print(f"  {dim_name}: {dict(sorted(dist.items()))}")
    print(f"  Cities: {len(set(p.get('city','') for p in selected))} unique")


if __name__ == "__main__":
    main()
