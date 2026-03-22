"""
Bias audit — measures cognitive biases in SGO's LLM evaluator pipeline.

Inspired by CoBRA (Liu et al., CHI'26 Best Paper, arXiv:2509.13588), this script
runs validated social science experiments through SGO's evaluation pipeline to
quantify how much bias the LLM evaluators exhibit.

This is the first step toward expert panel fidelity: you can't calibrate what
you can't measure.

Supported probes:
  - framing: same entity, gain vs. loss framing → measures framing effect
  - authority: entity with/without authority signals → measures authority bias
  - order: same entity, sections reordered → measures anchoring/order effects

Usage:
    uv run python scripts/bias_audit.py \
      --entity entities/my_product.md \
      --cohort data/cohort.json \
      --probes framing authority order \
      --sample 10 \
      --parallel 5

Output: results/bias_audit/report.md + raw data
"""

import json
import os
import re
import time
import argparse
import concurrent.futures
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI


# ── Evaluation core (reused from evaluate.py) ────────────────────────────

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


def evaluate_one(client, model, evaluator, entity_text):
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=16384,
            temperature=0.7,
        )
        content = resp.choices[0].message.content
        if not content:
            return {"error": "Empty response"}
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        result = json.loads(content)
        result["_evaluator"] = {
            "name": evaluator["name"],
            "age": evaluator.get("age"),
            "occupation": evaluator.get("occupation"),
        }
        return result
    except Exception as e:
        return {"error": str(e), "_evaluator": {"name": evaluator.get("name", "?")}}


# ── Bias probes ──────────────────────────────────────────────────────────

REFRAME_PROMPT = """You are a text transformation tool. Rewrite the following entity description
using {frame_type} framing. Keep ALL factual content identical — same features, same pricing,
same capabilities. Only change the rhetorical framing.

{frame_instruction}

Return the rewritten text only, no commentary.

---

{entity}"""

FRAME_INSTRUCTIONS = {
    "gain": "Emphasize what the user GAINS: benefits, improvements, positive outcomes. "
            'Use phrases like "save", "gain", "achieve", "unlock", "improve".',
    "loss": "Emphasize what the user LOSES without this: risks, costs of inaction, missed opportunities. "
            'Use phrases like "avoid losing", "stop wasting", "don\'t miss", "risk of", "falling behind".',
}


def reframe_entity(client, model, entity_text, frame_type):
    """Rewrite entity with gain or loss framing, preserving factual content."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": REFRAME_PROMPT.format(
            frame_type=frame_type,
            frame_instruction=FRAME_INSTRUCTIONS[frame_type],
            entity=entity_text,
        )}],
        max_tokens=16384,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


AUTHORITY_SIGNALS = [
    "Trusted by 10,000+ teams worldwide.",
    "SOC 2 Type II certified. GDPR compliant.",
    "Featured in TechCrunch, Wired, and Forbes.",
    "Backed by Sequoia Capital and Y Combinator.",
    "Winner of Product Hunt's Product of the Year.",
]


def add_authority_signals(entity_text):
    """Add authority/credibility signals to an entity."""
    signals = "\n".join(f"- {s}" for s in AUTHORITY_SIGNALS)
    return f"{entity_text}\n\n---\n\n### Trust & Recognition\n\n{signals}\n"


def reorder_entity(entity_text):
    """Reverse the order of sections in the entity document."""
    sections = re.split(r'\n(?=##?\s)', entity_text)
    if len(sections) <= 1:
        # Try splitting on blank lines if no headers
        sections = re.split(r'\n\n+', entity_text)

    if len(sections) <= 1:
        return entity_text  # Can't reorder a single section

    # Keep first section (title/intro), reverse the rest
    return sections[0] + "\n\n" + "\n\n".join(reversed(sections[1:]))


# ── Probe runners ────────────────────────────────────────────────────────

def run_paired_evaluation(client, model, evaluators, entity_a, entity_b, label_a, label_b, parallel):
    """Run the same cohort against two entity variants and compute deltas."""
    results = []

    def worker(ev):
        r_a = evaluate_one(client, model, ev, entity_a)
        r_b = evaluate_one(client, model, ev, entity_b)
        return {
            "evaluator": ev["name"],
            "age": ev.get("age"),
            "occupation": ev.get("occupation"),
            f"score_{label_a}": r_a.get("score"),
            f"score_{label_b}": r_b.get("score"),
            "delta": (r_b.get("score", 0) or 0) - (r_a.get("score", 0) or 0),
            f"reasoning_{label_a}": r_a.get("reasoning", ""),
            f"reasoning_{label_b}": r_b.get("reasoning", ""),
            "error": r_a.get("error") or r_b.get("error"),
        }

    done = [0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(worker, ev): ev for ev in evaluators}
        for fut in concurrent.futures.as_completed(futs):
            result = fut.result()
            results.append(result)
            done[0] += 1
            if result.get("error"):
                print(f"  [{done[0]}/{len(evaluators)}] {result['evaluator']}: ERROR")
            else:
                print(f"  [{done[0]}/{len(evaluators)}] {result['evaluator']}: "
                      f"{label_a}={result[f'score_{label_a}']} "
                      f"{label_b}={result[f'score_{label_b}']} "
                      f"Δ={result['delta']:+d}")

    return results


def run_framing_probe(client, model, evaluators, entity_text, parallel):
    """Framing Effect probe: gain-framed vs. loss-framed entity."""
    print("\n── Framing Effect Probe ──")
    print("Generating gain-framed and loss-framed variants...")

    gain_entity = reframe_entity(client, model, entity_text, "gain")
    loss_entity = reframe_entity(client, model, entity_text, "loss")

    return run_paired_evaluation(
        client, model, evaluators, gain_entity, loss_entity,
        "gain", "loss", parallel,
    ), {"gain_entity": gain_entity, "loss_entity": loss_entity}


def run_authority_probe(client, model, evaluators, entity_text, parallel):
    """Authority Bias probe: entity with vs. without authority signals."""
    print("\n── Authority Bias Probe ──")

    entity_with_authority = add_authority_signals(entity_text)

    return run_paired_evaluation(
        client, model, evaluators, entity_text, entity_with_authority,
        "baseline", "authority", parallel,
    ), {"entity_with_authority": entity_with_authority}


def run_order_probe(client, model, evaluators, entity_text, parallel):
    """Order Effect probe: original vs. reordered entity."""
    print("\n── Order Effect Probe ──")

    reordered = reorder_entity(entity_text)

    return run_paired_evaluation(
        client, model, evaluators, entity_text, reordered,
        "original", "reordered", parallel,
    ), {"reordered_entity": reordered}


# ── Analysis ─────────────────────────────────────────────────────────────

HUMAN_BASELINES = {
    "framing": {
        "description": "Tversky & Kahneman (1981): ~30% of subjects shift preference based on framing",
        "expected_shift_pct": 30,
    },
    "authority": {
        "description": "Milgram (1963): 65% obedience rate under authority pressure",
        "expected_shift_pct": 20,  # Conservative estimate for evaluation context
    },
    "order": {
        "description": "Primacy/recency effects: ideally 0% shift (order shouldn't matter)",
        "expected_shift_pct": 0,
    },
}


def analyze_probe(results, probe_name, label_a, label_b):
    """Analyze a probe's results and compare to human baselines."""
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {"probe": probe_name, "error": "No valid results"}

    deltas = [r["delta"] for r in valid]
    abs_deltas = [abs(d) for d in deltas]
    shifted = [r for r in valid if r["delta"] != 0]
    positive_shift = [r for r in valid if r["delta"] > 0]
    negative_shift = [r for r in valid if r["delta"] < 0]

    n = len(valid)
    avg_delta = sum(deltas) / n
    avg_abs_delta = sum(abs_deltas) / n
    shift_pct = 100 * len(shifted) / n
    baseline = HUMAN_BASELINES.get(probe_name, {})

    return {
        "probe": probe_name,
        "n": n,
        "avg_delta": round(avg_delta, 2),
        "avg_abs_delta": round(avg_abs_delta, 2),
        "max_delta": max(deltas),
        "min_delta": min(deltas),
        "shifted_pct": round(shift_pct, 1),
        "positive_shifts": len(positive_shift),
        "negative_shifts": len(negative_shift),
        "no_change": n - len(shifted),
        "human_baseline": baseline,
        "comparison": label_a + " vs " + label_b,
    }


def generate_report(all_analyses, model):
    """Generate the bias audit report."""
    lines = [
        "# SGO Bias Audit Report",
        f"\n**Date**: {datetime.now().isoformat()}",
        f"**Model**: {model}",
        f"**Method**: CoBRA-inspired social science experiments (arXiv:2509.13588)",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"{'Probe':<12} {'N':>4} {'Avg Δ':>7} {'|Δ|':>5} {'Shifted%':>9}  {'Human Baseline':>15}  Gap",
        "-" * 75,
    ]

    for a in all_analyses:
        if "error" in a:
            lines.append(f"{a['probe']:<12}  ERROR: {a['error']}")
            continue
        baseline_pct = a["human_baseline"].get("expected_shift_pct", "?")
        gap = ""
        if isinstance(baseline_pct, (int, float)):
            diff = a["shifted_pct"] - baseline_pct
            gap = f"{diff:+.1f}pp"
        lines.append(
            f"{a['probe']:<12} {a['n']:>4} {a['avg_delta']:>+6.2f} {a['avg_abs_delta']:>5.2f}"
            f"   {a['shifted_pct']:>5.1f}%  {str(baseline_pct)+('%' if isinstance(baseline_pct, (int,float)) else ''):>15}  {gap}"
        )

    lines.extend(["", "---", "", "## Interpretation", ""])

    for a in all_analyses:
        if "error" in a:
            continue

        lines.append(f"### {a['probe'].title()} Effect ({a['comparison']})")
        lines.append("")

        baseline = a["human_baseline"]
        if baseline:
            lines.append(f"**Human baseline**: {baseline.get('description', 'N/A')}")

        lines.append(f"**LLM result**: {a['shifted_pct']:.1f}% of evaluators shifted scores "
                     f"(avg |Δ| = {a['avg_abs_delta']:.2f} points)")

        expected = baseline.get("expected_shift_pct")
        if isinstance(expected, (int, float)):
            if a["shifted_pct"] > expected + 10:
                lines.append(f"**Assessment**: OVER-BIASED — LLM evaluators show more {a['probe']} "
                           f"sensitivity than humans. Consider adding de-biasing instructions.")
            elif a["shifted_pct"] < expected - 10:
                lines.append(f"**Assessment**: UNDER-BIASED — LLM evaluators show less {a['probe']} "
                           f"sensitivity than humans. The panel may be too rational.")
            else:
                lines.append(f"**Assessment**: WELL-CALIBRATED — within ±10pp of human baseline.")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Next Steps",
        "",
        "1. **If over-biased**: Add bias-awareness instructions to the evaluation prompt",
        "2. **If under-biased**: Consider if this is acceptable (more rational) or needs calibration",
        "3. **For order effects**: Any non-zero shift indicates entity structure matters — "
        "standardize entity format or average across orderings",
        "4. **Re-run after calibration**: Use this script to verify improvements",
        "",
        "## References",
        "",
        "- Liu, X., Shang, H., & Jin, H. (2025). CoBRA. arXiv:2509.13588 (CHI'26 Best Paper)",
        "- Tversky, A. & Kahneman, D. (1981). The framing of decisions. Science, 211(4481).",
        "- Milgram, S. (1963). Behavioral Study of Obedience. JASP, 67(4).",
    ])

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bias audit for SGO evaluator pipeline")
    parser.add_argument("--entity", required=True, help="Path to entity document")
    parser.add_argument("--cohort", default="data/cohort.json")
    parser.add_argument("--probes", nargs="+", default=["framing", "authority", "order"],
                        choices=["framing", "authority", "order"])
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of evaluators to sample for audit (smaller = faster)")
    parser.add_argument("--parallel", type=int, default=5)
    args = parser.parse_args()

    entity_text = Path(args.entity).read_text()

    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    model = os.getenv("LLM_MODEL_NAME")

    with open(args.cohort) as f:
        cohort = json.load(f)

    # Sample a subset for the audit (bias audit is 2x cost per evaluator per probe)
    import random
    random.seed(42)
    if args.sample and args.sample < len(cohort):
        evaluators = random.sample(cohort, args.sample)
    else:
        evaluators = cohort

    print(f"Bias Audit | {len(evaluators)} evaluators | Model: {model}")
    print(f"Probes: {', '.join(args.probes)}")

    probe_runners = {
        "framing": lambda: run_framing_probe(client, model, evaluators, entity_text, args.parallel),
        "authority": lambda: run_authority_probe(client, model, evaluators, entity_text, args.parallel),
        "order": lambda: run_order_probe(client, model, evaluators, entity_text, args.parallel),
    }

    all_results = {}
    all_analyses = []

    for probe_name in args.probes:
        t0 = time.time()
        results, metadata = probe_runners[probe_name]()
        elapsed = time.time() - t0

        label_a, label_b = {
            "framing": ("gain", "loss"),
            "authority": ("baseline", "authority"),
            "order": ("original", "reordered"),
        }[probe_name]

        analysis = analyze_probe(results, probe_name, label_a, label_b)
        analysis["elapsed_s"] = round(elapsed, 1)
        all_analyses.append(analysis)

        all_results[probe_name] = {
            "results": results,
            "metadata": metadata,
            "analysis": analysis,
        }

        print(f"\n  {probe_name}: avg Δ={analysis.get('avg_delta', '?'):+.2f}, "
              f"shifted={analysis.get('shifted_pct', '?')}%, "
              f"time={elapsed:.1f}s")

    # Save
    out_dir = PROJECT_ROOT / "results" / "bias_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw data
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            "results": v["results"],
            "analysis": v["analysis"],
        }
    with open(out_dir / "raw_data.json", "w") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # Report
    report = generate_report(all_analyses, model)
    with open(out_dir / "report.md", "w") as f:
        f.write(report)

    print(f"\nReport: {out_dir / 'report.md'}")
    print(f"Data:   {out_dir / 'raw_data.json'}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
