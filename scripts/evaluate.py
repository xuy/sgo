"""
f(θ, x) evaluator — scores an entity against an evaluator cohort.

The LLM inhabits each evaluator's persona and produces a structured assessment
of the entity. Domain-agnostic: the system prompt adapts to the entity type.

Usage:
    uv run python scripts/evaluate.py \
      --entity entities/my_product.md \
      --cohort data/cohort.json \
      --tag baseline \
      --parallel 5
"""

import json
import os
import re
import time
import argparse
import concurrent.futures
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI


SYSTEM_PROMPT = """You are an evaluation simulator. You will be given:
1. A detailed persona — a person with specific values, needs, context, and perspective
2. An entity to evaluate (a product, profile, proposal, pitch, resume, etc.)

Your job: fully inhabit this persona's perspective and evaluate the entity AS THEY WOULD.

Be honest and realistic. Not everything is a match. Consider:
- Their specific needs, budget, constraints, and priorities
- Whether this entity solves a real problem for them
- Trust signals and red flags from their perspective
- Practical fit with their situation
- What they'd compare this against

You MUST respond with valid JSON only."""

# Optional bias-aware addendum, appended to SYSTEM_PROMPT when --bias-calibration is used.
# Inspired by CoBRA (Liu et al., CHI'26, arXiv:2509.13588).
BIAS_CALIBRATION_ADDENDUM = """

Important evaluation guidelines for realistic assessment:
- Evaluate the SUBSTANCE of the entity, not its rhetorical framing. A gain-framed
  description ("save 30%") and a loss-framed description ("stop wasting 30%") should
  receive similar scores if the underlying value is the same.
- Weight authority signals (certifications, press mentions, investor logos) proportionally
  to how much this persona's real-world counterpart would actually verify and value them.
- The ORDER in which information appears should not affect your score. Evaluate the
  complete picture, not just first impressions.
- Real people have genuine cognitive biases — you should too. But calibrate to realistic
  human levels, not LLM defaults. A credential matters, but it's not everything."""

EVAL_PROMPT = """## Evaluator Persona

Name: {name}
Age: {age}
Location: {city}, {state}
Education: {education_level}
Occupation: {occupation}
Status: {marital_status}

{persona}

---

## Entity to Evaluate

{entity}

---

## Task

Inhabit {name}'s perspective completely. Evaluate this entity as they would.

Return JSON:
{{
    "score": <1-10, where 1=strong reject, 5=ambivalent, 10=enthusiastic yes>,
    "action": "<positive | neutral | negative>",
    "attractions": ["<what works for them, max 3>"],
    "concerns": ["<what gives them pause, max 3>"],
    "dealbreakers": ["<hard no's if any, empty list if none>"],
    "summary": "<1-2 sentences — how they'd describe this to a peer>",
    "reasoning": "<2-3 sentence internal monologue>"
}}"""


def evaluate_one(client, model, evaluator, entity_text, system_prompt=None):
    prompt = EVAL_PROMPT.format(
        name=evaluator["name"],
        age=evaluator.get("age", ""),
        city=evaluator.get("city", ""),
        state=evaluator.get("state", ""),
        education_level=evaluator.get("education_level", ""),
        occupation=evaluator.get("occupation", ""),
        marital_status=evaluator.get("marital_status", ""),
        persona=evaluator.get("persona", ""),
        entity=entity_text,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=16384,
            temperature=0.7,
        )
        content = resp.choices[0].message.content
        if not content:
            return {"error": f"Empty (finish_reason={resp.choices[0].finish_reason})"}
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        result = json.loads(content)
        result["_evaluator"] = {
            "name": evaluator["name"],
            "age": evaluator.get("age"),
            "city": evaluator.get("city"),
            "state": evaluator.get("state"),
            "education_level": evaluator.get("education_level"),
            "occupation": evaluator.get("occupation"),
            "marital_status": evaluator.get("marital_status"),
        }
        return result
    except Exception as e:
        return {"error": str(e), "_evaluator": {"name": evaluator.get("name", "?")}}


def analyze(results):
    valid = [r for r in results if "score" in r]
    if not valid:
        return "No valid results."

    scores = [r["score"] for r in valid]
    n = len(valid)
    actions = [r["action"] for r in valid]

    lines = [f"## Summary ({n} evaluated)\n"]
    lines.append(f"Average score: {sum(scores)/n:.1f}/10")
    for act in ("positive", "neutral", "negative"):
        c = actions.count(act)
        lines.append(f"  {act}: {c} ({100*c//n}%)")

    lines.append("\n### Top Attractions")
    all_a = [a for r in valid for a in r.get("attractions", [])]
    for a, c in Counter(all_a).most_common(8):
        lines.append(f"  [{c}x] {a}")

    lines.append("\n### Top Concerns")
    all_c = [c for r in valid for c in r.get("concerns", [])]
    for c, cnt in Counter(all_c).most_common(8):
        lines.append(f"  [{cnt}x] {c}")

    lines.append("\n### Dealbreakers")
    all_d = [d for r in valid for d in r.get("dealbreakers", [])]
    if all_d:
        for d, cnt in Counter(all_d).most_common(8):
            lines.append(f"  [{cnt}x] {d}")
    else:
        lines.append("  (none)")

    sorted_v = sorted(valid, key=lambda r: r["score"], reverse=True)
    lines.append("\n### Most Receptive (top 5)")
    for r in sorted_v[:5]:
        e = r["_evaluator"]
        lines.append(f"  {e['name']}, {e.get('age','')}, {e.get('occupation','')}")
        lines.append(f"    {r['score']}/10 — \"{r.get('summary','')}\"")

    lines.append("\n### Least Receptive (bottom 5)")
    for r in sorted_v[-5:]:
        e = r["_evaluator"]
        lines.append(f"  {e['name']}, {e.get('age','')}, {e.get('occupation','')}")
        lines.append(f"    {r['score']}/10 — \"{r.get('summary','')}\"")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="Path to entity document")
    parser.add_argument("--cohort", default="data/cohort.json")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--bias-calibration", action="store_true",
                        help="Add CoBRA-inspired bias calibration instructions (arXiv:2509.13588)")
    args = parser.parse_args()

    entity_text = Path(args.entity).read_text()

    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    model = os.getenv("LLM_MODEL_NAME")

    with open(args.cohort) as f:
        cohort = json.load(f)
    if args.limit:
        cohort = cohort[:args.limit]

    sys_prompt = SYSTEM_PROMPT
    if args.bias_calibration:
        sys_prompt += BIAS_CALIBRATION_ADDENDUM
        print("Bias calibration: ON (CoBRA-inspired, arXiv:2509.13588)")

    print(f"Evaluating {len(cohort)} evaluators | Model: {model} | Workers: {args.parallel}")

    results = [None] * len(cohort)
    done = [0]
    t0 = time.time()

    def worker(idx, ev):
        return idx, evaluate_one(client, model, ev, entity_text, system_prompt=sys_prompt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(worker, i, e): i for i, e in enumerate(cohort)}
        for fut in concurrent.futures.as_completed(futs):
            idx, result = fut.result()
            results[idx] = result
            done[0] += 1
            ev = result.get("_evaluator", {})
            score = result.get("score", "?")
            action = result.get("action", "?")
            icon = {"positive": "✅", "neutral": "🤔", "negative": "❌"}.get(action, "?")
            if "error" in result:
                print(f"  [{done[0]}/{len(cohort)}] {ev.get('name','?')}: ERROR")
            else:
                print(f"  [{done[0]}/{len(cohort)}] {ev.get('name','?')}: {icon} {action} ({score}/10)")

    print(f"\nDone in {time.time()-t0:.1f}s")

    # Save
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    analysis_text = analyze(results)
    with open(out_dir / "analysis.md", "w") as f:
        f.write(f"# Evaluation: {tag}\n\n")
        f.write(f"- **Entity**: {args.entity}\n")
        f.write(f"- **Cohort**: {args.cohort} ({len(results)} evaluators)\n")
        f.write(f"- **Model**: {model}\n")
        f.write(f"- **Date**: {datetime.now().isoformat()}\n\n")
        f.write(analysis_text)

    meta = {
        "tag": tag, "entity": args.entity, "cohort": args.cohort,
        "model": model, "cohort_size": len(results),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults:  {out_dir / 'raw_results.json'}")
    print(f"Analysis: {out_dir / 'analysis.md'}")
    print(f"\n{analysis_text}")


if __name__ == "__main__":
    main()
