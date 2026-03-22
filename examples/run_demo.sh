#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# SGO End-to-End Demo — CodeReview AI
#
# Demonstrates the full pipeline: entity → cohort → evaluate → counterfactual
# probe → bias audit → bias-calibrated re-evaluation.
#
# Prerequisites:
#   1. cd <sgo-root> && uv sync
#   2. cp .env.example .env  (fill in your LLM API key)
#   3. uv run python scripts/setup_data.py  (download Nemotron personas, once)
#
# Usage:
#   cd <sgo-root>
#   bash examples/run_demo.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SGO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SGO_DIR"

ENTITY="examples/entity_codereview_ai.md"
CHANGES="examples/changes_codereview_ai.json"
COHORT="data/demo_cohort.json"
TAG="demo_baseline"
TAG_CAL="demo_calibrated"
SAMPLE=50
AUDIT_SAMPLE=10
PARALLEL=5

echo "═══════════════════════════════════════════════════════════════"
echo "  SGO End-to-End Demo: CodeReview AI"
echo "═══════════════════════════════════════════════════════════════"

# ── Phase 1: Entity already exists at examples/entity_codereview_ai.md ───

echo ""
echo "Phase 1 — Entity: $ENTITY"
echo "─────────────────────────────────────────────────────────────"
head -3 "$ENTITY"
echo "..."
echo ""

# ── Phase 2: Build cohort ────────────────────────────────────────────────

echo "Phase 2 — Building evaluator cohort ($SAMPLE personas)"
echo "─────────────────────────────────────────────────────────────"

# Filter: US adults 25-55 to get a broad software buyer population
uv run python scripts/persona_loader.py \
  --filters '{"age_min": 25, "age_max": 55}' \
  --output data/demo_filtered.json

# Stratified sample with entity-aware occupation bucketing
uv run python scripts/stratified_sampler.py \
  --input data/demo_filtered.json \
  --entity "$ENTITY" \
  --total "$SAMPLE" \
  --output "$COHORT"

echo ""

# ── Phase 3: Evaluate (baseline, no bias calibration) ───────────────────

echo "Phase 3 — Evaluating (baseline, no bias calibration)"
echo "─────────────────────────────────────────────────────────────"

uv run python scripts/evaluate.py \
  --entity "$ENTITY" \
  --cohort "$COHORT" \
  --tag "$TAG" \
  --parallel "$PARALLEL"

echo ""

# ── Phase 4: Counterfactual probe ────────────────────────────────────────

echo "Phase 4 — Counterfactual probe (semantic gradient)"
echo "─────────────────────────────────────────────────────────────"

uv run python scripts/counterfactual.py \
  --tag "$TAG" \
  --changes "$CHANGES" \
  --parallel "$PARALLEL"

echo ""

# ── Phase 6: Bias audit ─────────────────────────────────────────────────

echo "Phase 6 — Bias Audit (CoBRA-inspired, arXiv:2509.13588)"
echo "─────────────────────────────────────────────────────────────"
echo "Running framing, authority, and order probes on $AUDIT_SAMPLE evaluators..."

uv run python scripts/bias_audit.py \
  --entity "$ENTITY" \
  --cohort "$COHORT" \
  --probes framing authority order \
  --sample "$AUDIT_SAMPLE" \
  --parallel "$PARALLEL"

echo ""

# ── Phase 3 (re-run): Evaluate with bias calibration ────────────────────

echo "Phase 3 (re-run) — Evaluating with --bias-calibration"
echo "─────────────────────────────────────────────────────────────"

uv run python scripts/evaluate.py \
  --entity "$ENTITY" \
  --cohort "$COHORT" \
  --tag "$TAG_CAL" \
  --bias-calibration \
  --parallel "$PARALLEL"

echo ""

# ── Phase 5: Compare baseline vs. calibrated ────────────────────────────

echo "Phase 5 — Comparing baseline vs. bias-calibrated"
echo "─────────────────────────────────────────────────────────────"

uv run python scripts/compare.py --runs "$TAG" "$TAG_CAL"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Demo complete!"
echo ""
echo "  Results:"
echo "    Baseline:     results/$TAG/analysis.md"
echo "    Gradient:     results/$TAG/counterfactual/gradient.md"
echo "    Bias audit:   results/bias_audit/report.md"
echo "    Calibrated:   results/$TAG_CAL/analysis.md"
echo "═══════════════════════════════════════════════════════════════"
