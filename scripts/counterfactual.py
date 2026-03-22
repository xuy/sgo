"""
Counterfactual probe — semantic gradient estimation.

Takes evaluation results, identifies the movable middle, and asks the LLM to
estimate score deltas for hypothetical changes. Produces a Jacobian matrix
and aggregated gradient.

Usage:
    uv run python scripts/counterfactual.py \
      --tag baseline \
      --changes data/changes.json \
      --parallel 5
"""

import json
import os
import re
import time
import argparse
import concurrent.futures
from collections import defaultdict, Counter
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI


SYSTEM_PROMPT = """You are performing counterfactual analysis on a prior evaluation.

You previously evaluated an entity from a specific persona's perspective and gave a score.
Now estimate how SPECIFIC CHANGES to the entity would shift that score.

Rules:
- Stay fully in character as this persona
- Be realistic — some changes matter a lot, others barely register
- A change can be positive, negative, or neutral depending on this persona's values
- Consider second-order effects
- Score deltas reflect THIS persona's specific perspective

You MUST respond with valid JSON only."""


PROBE_PROMPT = """## Evaluator Persona

Name: {name}
Age: {age}
Location: {city}, {state}
Occupation: {occupation}

{persona}

## Their Original Evaluation

Score: {original_score}/10, Action: {original_action}
Reasoning: "{original_reasoning}"
Concerns: {original_concerns}

## Counterfactual Changes

For each change below, estimate the NEW score (1-10) if this change were applied.

{changes_block}

Return JSON:
{{
    "original_score": {original_score},
    "counterfactuals": [
        {{
            "change_id": "<id>",
            "new_score": <1-10>,
            "delta": <new minus original>,
            "impact": "<high | medium | low | none | negative>",
            "reasoning": "<1 sentence — why this matters or doesn't to THEM>"
        }}
    ]
}}"""


def build_changes_block(changes):
    lines = []
    for i, c in enumerate(changes, 1):
        lines.append(f"### Change {i}: {c['label']} (id: {c['id']})")
        lines.append(c["description"])
        lines.append("")
    return "\n".join(lines)


def probe_one(client, model, eval_result, cohort_map, all_changes):
    ev = eval_result.get("_evaluator", {})
    name = ev.get("name", "")
    persona_text = cohort_map.get(name, {}).get("persona", "")

    prompt = PROBE_PROMPT.format(
        name=name, age=ev.get("age", ""),
        city=ev.get("city", ""), state=ev.get("state", ""),
        occupation=ev.get("occupation", ""),
        persona=persona_text,
        original_score=eval_result["score"],
        original_action=eval_result.get("action", ""),
        original_reasoning=eval_result.get("reasoning", ""),
        original_concerns=json.dumps(eval_result.get("concerns", [])),
        changes_block=build_changes_block(all_changes),
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=16384,
            temperature=0.4,
        )
        content = resp.choices[0].message.content
        if not content:
            return {"error": "Empty response"}
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        result = json.loads(content)
        result["_evaluator"] = ev
        return result
    except Exception as e:
        return {"error": str(e), "_evaluator": ev}


GOAL_RELEVANCE_PROMPT = """You are scoring how relevant an evaluator is to a specific goal.

## Goal
{goal}

## Evaluator
Name: {name}, Age: {age}, Occupation: {occupation}
Their evaluation: {score}/10 — "{summary}"

## Task
On a scale of 0.0 to 1.0, how relevant is this evaluator's opinion to the stated goal?
- 1.0 = this is exactly the kind of person whose opinion matters for this goal
- 0.5 = somewhat relevant
- 0.0 = completely irrelevant to this goal

Return JSON only: {{"relevance": <0.0-1.0>, "reasoning": "<1 sentence>"}}"""


def compute_goal_weights(client, model, eval_results, cohort_map, goal, parallel=5):
    """Score each evaluator's relevance to the goal. Returns {name: weight}."""
    weights = {}

    def score_one(r):
        ev = r.get("_evaluator", {})
        name = ev.get("name", "")
        persona = cohort_map.get(name, {})
        prompt = GOAL_RELEVANCE_PROMPT.format(
            goal=goal, name=name, age=ev.get("age", ""),
            occupation=ev.get("occupation", ""),
            score=r.get("score", "?"),
            summary=r.get("summary", r.get("reasoning", "")),
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=256, temperature=0.3,
            )
            content = resp.choices[0].message.content
            content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
            data = json.loads(content)
            return name, float(data.get("relevance", 0.5)), data.get("reasoning", "")
        except Exception:
            return name, 0.5, "default"

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = [pool.submit(score_one, r) for r in eval_results]
        for fut in concurrent.futures.as_completed(futs):
            name, weight, reasoning = fut.result()
            weights[name] = {"weight": weight, "reasoning": reasoning}

    return weights


def analyze_gradient(results, all_changes, goal_weights=None):
    valid = [r for r in results if "counterfactuals" in r]
    if not valid:
        return "No valid results."

    has_goal = goal_weights is not None
    labels = {c["id"]: c["label"] for c in all_changes}
    jacobian = defaultdict(list)

    for r in valid:
        name = r["_evaluator"].get("name", "")
        w = goal_weights.get(name, {}).get("weight", 1.0) if has_goal else 1.0
        for cf in r.get("counterfactuals", []):
            jacobian[cf.get("change_id", "")].append({
                "delta": cf.get("delta", 0),
                "weighted_delta": cf.get("delta", 0) * w,
                "weight": w,
                "name": name,
                "age": r["_evaluator"].get("age", ""),
                "reasoning": cf.get("reasoning", ""),
            })

    ranked = []
    for cid, deltas in jacobian.items():
        total_weight = sum(d["weight"] for d in deltas)
        if total_weight == 0:
            total_weight = 1
        weighted_avg = sum(d["weighted_delta"] for d in deltas) / total_weight
        raw_avg = sum(d["delta"] for d in deltas) / len(deltas)
        ranked.append({
            "id": cid, "label": labels.get(cid, cid),
            "avg_delta": weighted_avg,
            "raw_avg_delta": raw_avg,
            "max_delta": max(d["delta"] for d in deltas),
            "min_delta": min(d["delta"] for d in deltas),
            "positive": sum(1 for d in deltas if d["delta"] > 0),
            "negative": sum(1 for d in deltas if d["delta"] < 0),
            "n": len(deltas), "details": deltas,
        })
    ranked.sort(key=lambda x: x["avg_delta"], reverse=True)

    mode = "Goal-Weighted (VJP)" if has_goal else "Uniform"
    lines = [f"# Semantic Gradient ({mode})\n\nProbed {len(valid)} evaluators across {len(all_changes)} changes.\n"]
    if has_goal:
        header = f"{'Rank':<5} {'VJP Δ':>6} {'Raw Δ':>6} {'Max':>5} {'Min':>5}  Change"
    else:
        header = f"{'Rank':<5} {'Avg Δ':>6} {'Max':>5} {'Min':>5} {'👍':>4} {'👎':>4}  Change"
    lines.append(header)
    lines.append("-" * 75)
    for i, r in enumerate(ranked, 1):
        if has_goal:
            lines.append(
                f"{i:<5} {r['avg_delta']:>+5.1f}  {r['raw_avg_delta']:>+5.1f}  "
                f"{r['max_delta']:>+4}  {r['min_delta']:>+4}   {r['label']}"
            )
        else:
            lines.append(
                f"{i:<5} {r['avg_delta']:>+5.1f}  {r['max_delta']:>+4}  {r['min_delta']:>+4}  "
                f"{r['positive']:>3}  {r['negative']:>3}   {r['label']}"
            )

    lines.append(f"\n## Top 3 — Detail\n")
    for r in ranked[:3]:
        label = f"### {r['label']} (Δ {r['avg_delta']:+.1f})"
        if has_goal and abs(r['avg_delta'] - r['raw_avg_delta']) > 0.2:
            label += f"  ← was {r['raw_avg_delta']:+.1f} without goal weighting"
        lines.append(label + "\n")
        positive = sorted([d for d in r["details"] if d["delta"] > 0],
                          key=lambda x: x["weighted_delta"] if has_goal else x["delta"],
                          reverse=True)
        if positive:
            lines.append("**Helps:**")
            for d in positive[:5]:
                w_label = f" [w={d['weight']:.1f}]" if has_goal else ""
                lines.append(f"  +{d['delta']} {d['name']} ({d['age']}){w_label}: {d['reasoning']}")
        negative = [d for d in r["details"] if d["delta"] < 0]
        if negative:
            lines.append("**Hurts:**")
            for d in sorted(negative, key=lambda x: x["delta"])[:3]:
                w_label = f" [w={d['weight']:.1f}]" if has_goal else ""
                lines.append(f"  {d['delta']} {d['name']} ({d['age']}){w_label}: {d['reasoning']}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--changes", required=True, help="JSON file with changes to probe")
    parser.add_argument("--goal", default=None,
                        help="Goal to optimize toward (enables VJP weighting)")
    parser.add_argument("--min-score", type=int, default=4)
    parser.add_argument("--max-score", type=int, default=7)
    parser.add_argument("--parallel", type=int, default=5)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "results" / args.tag
    with open(run_dir / "raw_results.json") as f:
        eval_results = json.load(f)
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    with open(meta.get("cohort", "data/cohort.json")) as f:
        cohort = json.load(f)
    with open(args.changes) as f:
        changes_data = json.load(f)

    # Support both flat list and categorized dict
    if isinstance(changes_data, list):
        all_changes = changes_data
    else:
        all_changes = []
        for cat in changes_data.values():
            all_changes.extend(cat if isinstance(cat, list) else cat.get("changes", []))

    cohort_map = {p["name"]: p for p in cohort}

    movable = [r for r in eval_results
                if "score" in r and args.min_score <= r["score"] <= args.max_score]

    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    model = os.getenv("LLM_MODEL_NAME")

    print(f"Movable middle (score {args.min_score}-{args.max_score}): {len(movable)}")
    print(f"Changes: {len(all_changes)} | Model: {model}")
    if args.goal:
        print(f"Goal: {args.goal} (VJP mode)\n")
    else:
        print("No goal — uniform weighting\n")

    results = [None] * len(movable)
    done = [0]
    t0 = time.time()

    def worker(idx, r):
        return idx, probe_one(client, model, r, cohort_map, all_changes)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(worker, i, r): i for i, r in enumerate(movable)}
        for fut in concurrent.futures.as_completed(futs):
            idx, result = fut.result()
            results[idx] = result
            done[0] += 1
            ev = result.get("_evaluator", {})
            cfs = result.get("counterfactuals", [])
            top = max(cfs, key=lambda c: c.get("delta", 0)) if cfs else {}
            if "error" in result:
                print(f"  [{done[0]}/{len(movable)}] {ev.get('name','?')}: ERROR")
            else:
                print(f"  [{done[0]}/{len(movable)}] {ev.get('name','?')} "
                      f"(orig {result.get('original_score','?')}) "
                      f"best Δ: +{top.get('delta',0)} from '{top.get('change_id','?')}'")

    print(f"\nDone in {time.time()-t0:.1f}s")

    out_dir = run_dir / "counterfactual"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "raw_probes.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Compute goal weights if goal is specified (VJP)
    goal_weights = None
    if args.goal:
        print("Computing goal-relevance weights...")
        goal_weights = compute_goal_weights(
            client, model, eval_results, cohort_map, args.goal,
            parallel=args.parallel,
        )
        relevant = sum(1 for v in goal_weights.values() if v["weight"] >= 0.5)
        print(f"  {relevant}/{len(goal_weights)} evaluators relevant to goal\n")

    gradient = analyze_gradient(results, all_changes, goal_weights=goal_weights)
    with open(out_dir / "gradient.md", "w") as f:
        f.write(gradient)

    print(f"\nGradient: {out_dir / 'gradient.md'}")
    print(f"\n{gradient}")


if __name__ == "__main__":
    main()
