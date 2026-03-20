---
name: sgo
description: "Semantic Gradient Optimization — optimize any entity (product, resume, pitch, profile) against a population of evaluators using LLMs and counterfactual probes. Use when the user wants to evaluate how something they control would be perceived by a target audience, find what to change, or compare versions over time."
argument-hint: "[entity-path or description]"
---

# Semantic Gradient Optimization

You are executing the SGO pipeline. This optimizes an entity the user controls (θ) against a population of evaluators (x), using LLM-based scoring and counterfactual probes to estimate a semantic gradient.

**Repo**: https://github.com/xuy/sgo

## First: find the install location

The SGO repo could be installed at:
- `~/.claude/skills/sgo/` (if installed as a skill)
- Wherever the user cloned it

Run `ls ~/.claude/skills/sgo/scripts/evaluate.py 2>/dev/null` to check. If not found, ask the user where they cloned SGO, or offer to clone it:

```bash
git clone https://github.com/xuy/sgo.git ~/.claude/skills/sgo
```

For the rest of this file, `$SGO_DIR` refers to the SGO install location. All paths are relative to it.

## Quick Reference

```
Pipeline:  Build Entity → Build Cohort → Score f(θ,x) → Probe Counterfactuals → Act & Re-evaluate
Scripts:   $SGO_DIR/scripts/
Templates: $SGO_DIR/templates/
Data:      $SGO_DIR/data/
Results:   $SGO_DIR/results/
```

---

## Phase 0 — Setup

1. **Dependencies**: `cd $SGO_DIR && uv sync` (or `pip install datasets huggingface_hub openai python-dotenv`)
2. **API key**: Check `$SGO_DIR/.env` exists. If not, `cp $SGO_DIR/.env.example $SGO_DIR/.env` and ask the user to fill in their LLM API key. Do NOT read `.env` — just confirm it's configured.
3. **Persona data**: Run `uv run python $SGO_DIR/scripts/setup_data.py`. This downloads NVIDIA Nemotron-Personas-USA (~2GB) to `$SGO_DIR/data/nemotron/` on first run and skips if already cached.

---

## Phase 1 — Entity (θ)

Ask the user: **"What are you optimizing?"**

Options: product, resume, pitch, policy, profile, or custom.

- If they have a document, save it to `$SGO_DIR/entities/<name>.md`
- If not, use a template from `$SGO_DIR/templates/` and fill it in together
- Confirm: *"Here's what evaluators will see. Anything to add or remove?"*

---

## Phase 2 — Cohort ({xᵢ})

Ask: **"Who is your target audience?"** and **"What dimensions matter for segmentation?"**

Two paths:

**A) Nemotron dataset** (preferred — census-grounded, 1M US personas):
```bash
cd $SGO_DIR

# Filter
uv run python scripts/persona_loader.py \
  --filters '{"sex": "...", "state": "...", "age_min": N, "age_max": N}' \
  --output data/filtered.json

# Stratified sample
uv run python scripts/stratified_sampler.py \
  --input data/filtered.json \
  --total 50 \
  --output data/cohort.json
```

**B) LLM-generated** (fallback — warn user about mode collapse risk):
```bash
uv run python scripts/generate_cohort.py \
  --description "..." \
  --segments '[{"label": "...", "count": N}, ...]' \
  --output data/cohort.json
```

Show the cohort distribution table. Confirm with user.

---

## Phase 3 — Evaluate

```bash
cd $SGO_DIR
uv run python scripts/evaluate.py \
  --entity entities/<name>.md \
  --cohort data/cohort.json \
  --tag <run_tag> \
  --parallel 5
```

Present: avg score, breakdown by segment, top attractions, top concerns, dealbreakers, most/least receptive evaluators with quotes.

Ask: **"Anything surprising? Want to dig into a segment?"**

---

## Phase 4 — Counterfactual Probe

Ask: **"What changes are you considering?"** and **"What won't you change?"**

If unsure, propose changes based on top concerns from Phase 3. Save to `data/changes.json` (see `templates/changes.json` for format).

```bash
uv run python scripts/counterfactual.py \
  --tag <run_tag> \
  --changes data/changes.json \
  --parallel 5
```

Present the semantic gradient: ranked changes by avg Δ, % helped, % hurt, demographic sensitivity.

Ask: **"Which change do you want to make first?"**

---

## Phase 5 — Iterate

1. User updates entity → save as `entities/<name>_v2.md`
2. Re-evaluate with same cohort: `--tag <new_tag>`
3. Compare: `uv run python scripts/compare.py --runs <old> <new>`
4. Show delta, new attractions, resolved concerns
5. Ask: **"Another round, or are we good?"**

---

## Key Principles

- **Cohort is the control group** — keep it fixed across runs
- **Census-grounded > LLM-generated** personas (see README)
- **Only probe actionable changes** — don't waste tokens on things they can't or won't change
- **The gradient is semantic** — present the reasoning, not just the deltas
- **Each run is tagged** — results are comparable longitudinally

## Arguments

If `$ARGUMENTS` is provided:
- If it's a file path → use as the entity (skip Phase 1)
- If it's a description → use to draft the entity in Phase 1
- If it's a run tag → load existing results and continue from where they left off
