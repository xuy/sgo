# Semantic Gradient Optimization — Agent Instructions

You are executing the Semantic Gradient Optimization pipeline. This file tells you how to run it end-to-end, interacting with the user at each decision point.

Read `README.md` first for the full framework. This file is the execution guide.

---

## Phase 0 — Setup

### Check dependencies

```bash
cd <project_dir>
uv sync
```

If `uv` is not installed or `pyproject.toml` is missing, install dependencies manually:

```bash
pip install datasets huggingface_hub openai python-dotenv
```

### Check API key

The user needs an OpenAI-compatible LLM API key in `.env`:

```
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL_NAME=...
```

If `.env` doesn't exist, copy `.env.example` and ask the user to fill it in. Do NOT read the `.env` file — ask the user to confirm it's configured.

### Check data

If `~/Data/nvidia/Nemotron-Personas-USA/dataset_info.json` exists, the persona dataset is ready. If not, run:

```bash
uv run python scripts/setup_data.py
```

This downloads the 1M-persona dataset (~2GB). Only needs to happen once.

---

## Phase 1 — Define the Entity (θ)

**Ask the user**:

1. *"What are you optimizing? (product, resume, pitch, policy, dating profile, or describe your own)"*
2. *"Describe it — or paste/point me to the document. I need what an evaluator would see."*
3. *"Is there anything an evaluator should NOT see? (internal metrics, private details, etc.)"*

**Then**:

- Write the entity to `entities/<name>.md`
- Confirm with the user: *"Here's what I'll show evaluators. Anything to add or remove?"*

If the user doesn't have a document ready, use the appropriate template from `templates/` as a starting point and fill it in together.

---

## Phase 2 — Define the Evaluator Population

**Ask the user**:

1. *"Who evaluates this? Describe your target audience."*
   - Examples: "startup CTOs", "hiring managers at FAANG", "homeowners in the Bay Area"
2. *"What dimensions matter most for segmentation?"*
   - Suggest defaults based on the domain (see table below)
3. *"Do you have a persona dataset, or should I use Nemotron-Personas-USA?"*

### Default stratification dimensions by domain

| Domain | Suggested dimensions |
|--------|---------------------|
| Product | Company size, role, budget, tech stack, geography |
| Resume | Company type, seniority, technical depth, industry |
| Pitch | Investment stage, sector focus, check size |
| Policy | Stakeholder role, income bracket, geography, property ownership |
| Dating | Age bracket, life stage, education, occupation, geography |
| Custom | Ask the user to name 3-4 dimensions |

### Build the cohort

Run the stratified sampler with the user's parameters:

```bash
uv run python scripts/stratified_sampler.py \
  --population <dataset_or_generated> \
  --filters '{"sex": "Female", "state": "IL", "age_min": 25, "age_max": 50}' \
  --dimensions '["age_bracket", "marital_status", "education_tier"]' \
  --total 50 \
  --output data/cohort.json
```

If Nemotron doesn't fit the domain (e.g., evaluating a B2B product where you need CTO personas, not general population), generate personas using `scripts/generate_cohort.py` instead. But warn the user about the seeding quality difference (see README.md § The Seeding Problem).

**Confirm**: *"Here's the cohort: N evaluators across M strata. [show distribution table]. Look right?"*

---

## Phase 3 — Evaluate: f(θ, xᵢ)

Run the evaluation:

```bash
uv run python scripts/evaluate.py \
  --entity entities/<name>.md \
  --cohort data/cohort.json \
  --tag <run_tag> \
  --parallel 5
```

**Present results to the user**:

1. Overall score distribution (avg, swipe-right %, swipe-left %)
2. Breakdown by each stratification dimension
3. Top 5 attractions (aggregated)
4. Top 5 concerns (aggregated)
5. Any dealbreakers
6. Most and least interested evaluators (with quotes)

**Ask**: *"Any of these results surprising? Want to dig into a specific segment before we move to optimization?"*

---

## Phase 4 — Counterfactual Probe (Semantic Gradient)

### Generate candidate changes

**Ask the user**:

1. *"What changes are you considering? List anything — I'll categorize them."*
2. *"What will you NOT change? (boundaries/non-negotiables)"*

If the user isn't sure, propose changes based on the top concerns from Phase 3:

- For each top concern, generate 1-2 changes that would address it
- Categorize each as: presentation (free), actionable (has cost), fixed, or boundary
- Filter out fixed and boundary — only probe the first two

Write changes to `data/changes.json` or use defaults.

### Run the probe

```bash
uv run python scripts/counterfactual.py \
  --tag <run_tag> \
  --changes data/changes.json \
  --min-score 4 --max-score 7 \
  --parallel 5
```

**Present the semantic gradient**:

1. Priority-ranked table: change, avg Δ, % helped, % hurt
2. Top 3 changes with per-evaluator reasoning
3. Demographic sensitivity: which changes help which segments
4. Any changes that hurt certain segments (tradeoffs)

**Ask**: *"Based on this gradient, which change do you want to make first? Or should we test a compound change?"*

---

## Phase 5 — Iterate

Once the user makes a change:

1. Update the entity document: `entities/<name>_v2.md`
2. Re-run evaluation with the same cohort: `--tag <new_tag>`
3. Run comparison:

```bash
uv run python scripts/compare.py --runs <old_tag> <new_tag>
```

4. Present the delta: what improved, what regressed, concerns resolved, new concerns
5. Ask: *"Want to probe the next round of changes, or are we good?"*

Repeat until the user is satisfied or diminishing returns are clear.

---

## Decision Tree

```
Start
  │
  ▼
Has entity document?
  ├─ Yes → Phase 2
  └─ No  → Phase 1: build it together
  │
  ▼
Has evaluator cohort?
  ├─ Yes (from prior run) → reuse, go to Phase 3
  └─ No → Phase 2: define audience, build cohort
  │
  ▼
Has evaluation results?
  ├─ Yes (from prior run) → show summary, ask if re-run needed
  └─ No → Phase 3: run evaluation
  │
  ▼
User wants optimization?
  ├─ Yes → Phase 4: counterfactual probe
  └─ No  → done, save results
  │
  ▼
User made changes?
  ├─ Yes → Phase 5: re-evaluate, compare
  └─ No  → done
```

---

## File Layout

```
<project_dir>/
├── README.md              # Framework (for humans)
├── AGENT.md               # This file (for agents)
├── LICENSE
├── pyproject.toml
├── .env.example
├── scripts/
│   ├── setup_data.py      # Download Nemotron dataset
│   ├── persona_loader.py  # Load + filter personas
│   ├── stratified_sampler.py
│   ├── generate_cohort.py # LLM-generate personas when no dataset fits
│   ├── evaluate.py        # f(θ, x) scorer
│   ├── counterfactual.py  # Semantic gradient probe
│   └── compare.py         # Cross-run diff
├── templates/
│   ├── entity_product.md
│   ├── entity_resume.md
│   ├── entity_pitch.md
│   └── changes.json       # Default counterfactual template
├── entities/              # User's entity documents (θ)
├── data/                  # Cohorts, filtered datasets
└── results/               # One subdir per run tag
    └── <tag>/
        ├── meta.json
        ├── raw_results.json
        ├── analysis.md
        └── counterfactual/
            ├── raw_probes.json
            └── gradient.md
```
