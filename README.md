---
title: SGO — Semantic Gradient Optimization
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# SGO — Semantic Gradient Optimization

You're launching a product. You think the landing page is good. But **who have you actually asked?**

You could run a survey — but that takes weeks and you'd need to find the right people. You could ask an LLM — but one LLM opinion isn't a market. You could A/B test — but you need traffic first, and you don't know *what* to test.

**SGO lets you ask 50 realistic people what they think — in 3 minutes, for $0.10.**

It builds a representative panel from census-grounded synthetic personas, has each one score your thing from their perspective, then asks *"what would change your mind?"* — producing a priority-ranked list of what to fix first.

```
You: "Here's my landing page. Here's my target market."

SGO: "47 evaluators scored you. Avg 5.3/10.
      Solo devs love it (7.2). Enterprise is blocked (3.1).
      #1 concern: no SOC2. #2: no free tier.

      Gradient:
        +2.1  Add self-hosted option
        +1.8  Add free tier           ← biggest universal win
        +1.4  Get SOC2 certified
        +0.6  Drop price              ← not actually the blocker"
```

---

## What Can You Use It For?

Anything someone else evaluates.

| What you're optimizing | Who evaluates it | What you learn |
|----------------------|-----------------|---------------|
| **Product** — landing page, pricing | Buyer personas by company size, role, budget | Which segments convert, which are blocked, and why |
| **Resume** — CV + cover letter | Hiring managers at startups vs. enterprises | What stands out, what's a red flag, what to lead with |
| **Pitch** — investor deck | VCs and angels at different stages | Whether the story lands, what questions they'd ask |
| **Policy** — proposed regulation | Stakeholders by role, income, geography | Who supports it, who opposes, what compromise works |
| **Content** — blog post, video | Readers at different expertise levels | Whether it hits the right level, what's confusing |
| **Profile** — professional bio, personal brand | Population sample by age, education, occupation | How different demographics perceive you |

SGO ships with a 1M-person census-grounded dataset ([Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA)) with structured demographics (age, sex, education, occupation, marital status, US geography) plus rich narrative fields — professional persona, skills and expertise, career goals, hobbies, cultural background, and personality. The narratives naturally encode things like seniority, industry, technical depth, and decision-making style, even though those aren't separate columns.

This means most domains work out of the box — the LLM evaluates from the persona's full context, not just the demographic fields. For highly specialized panels (e.g., Series B VCs, enterprise procurement officers), SGO can generate personas via LLM with explicit stratification constraints. See [limitations](#limitations) on generated vs. census-grounded panels.

In each case, SGO tells you **where you stand**, **what's working**, **what's not**, and **what specific change would help the most** — broken down by audience segment.

---

## Quick Start

```bash
git clone https://github.com/xuy/sgo.git && cd sgo
cp .env.example .env   # Add your LLM API key (any OpenAI-compatible provider)
uv sync
uv run --extra web python web/app.py
# Opens at http://localhost:8000
```

The web interface walks you through the full pipeline: describe your entity, build a panel, evaluate, find the highest-impact changes, and audit your panel for cognitive biases.

<details>
<summary>Alternative: use as a Claude Code skill</summary>

```bash
git clone https://github.com/xuy/sgo.git ~/.claude/skills/sgo
cd ~/.claude/skills/sgo && cp .env.example .env && uv sync
```

Then run:

```
/sgo                              # Interactive — it asks what you're optimizing
/sgo entities/my_product.md       # Start with an existing entity
/sgo "optimize my landing page"   # Start from a description
```

</details>

<details>
<summary>CLI-only usage (no web interface)</summary>

```bash
uv run python scripts/setup_data.py   # Download Nemotron personas (once, ~2GB)
# Then use scripts directly: evaluate.py, counterfactual.py, bias_audit.py, compare.py
# See AGENT.md for the full pipeline reference
```

</details>

---

## How It Works

You describe what you're optimizing and what your goal is. SGO builds a diverse panel, has each one react, then focuses on the **persuadable middle** — the people who are *almost* convinced — to find what would tip them toward your goal.

SGO does **not** try to please everyone. People who scored 1–3 are not your audience — their feedback is informational, not actionable. The system focuses on moving the people who are close to yes.

**Five steps:**

1. **Describe your entity and goal** — what an evaluator would see, and what outcome you're optimizing for
2. **Build a panel** — 30–80 evaluators, stratified to cover the segments that matter
3. **Evaluate** — each evaluator scores 1–10. Results are segmented: champions (8+), persuadable (4–7), not-for-them (1–3)
4. **Find directions for your goal** — the persuadable middle re-evaluates hypothetical changes. With a goal, evaluators are weighted by relevance (VJP)
5. **Act and re-run** — make the top change, re-evaluate against the same panel, track improvement over time

The key insight is step 4. The probe produces a ranked list of changes sorted by how much they'd move the persuadable middle toward your goal. SGO calls this the **semantic gradient** — technically a vector-Jacobian product when a goal is specified.

<details>
<summary>Example: what the gradient looks like</summary>

Each row is an evaluator. Each column is a hypothetical change. Each cell is the score delta.

| | Add free tier | Get SOC2 | Self-hosted | Open-core | Case studies |
|---|:---:|:---:|:---:|:---:|:---:|
| Solo dev | +2 | +1 | 0 | +1 | +3 |
| Startup EM | +1 | +3 | -1 | +2 | +4 |
| Enterprise CTO | 0 | +1 | +2 | +1 | +2 |
| Data analyst | +1 | +2 | 0 | 0 | +3 |
| **Average** | **+1.0** | **+1.8** | **+0.3** | **+1.0** | **+3.0** |

The column averages tell you what to fix first. "Case studies" has the highest average impact. "Self-hosted" helps enterprise but slightly hurts startups — a tradeoff, not a pure win.

</details>

### What makes the panel realistic?

SGO uses [NVIDIA Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) — 1 million synthetic Americans whose demographics match real US census distributions. Each persona includes detailed narratives: professional background, skills, career goals, hobbies, cultural background, and personality.

This matters because when you ask an LLM to "generate 50 diverse personas," you get 5–6 archetypes with surface variation — mostly coastal, college-educated, and tech-adjacent. You can't audit what's missing. Census-grounded personas give you the construction worker in suburban Illinois and the quilter in rural Texas, because census data says those people exist.

The principle: **define the population before the measurement, not after.**

### From general population to any domain

Nemotron covers age, sex, education, occupation, geography, and marital status as structured fields — plus rich narratives about each person's career, skills, values, and lifestyle. That's enough to directly evaluate anything consumer-facing: products, profiles, content, policy.

But what about domains the dataset doesn't explicitly cover — like "enterprise CTOs" or "Series B investors"? There are four ways to get there, from most grounded to most flexible:

**1. Filter by what's already there.** A Nemotron persona with `occupation: software_developer`, `education: graduate`, `age: 38` and a professional narrative describing team leadership *is* a plausible engineering manager evaluating your developer tool. You just filter and let the narrative do the work.

**2. Reframe the evaluation prompt.** Same persona, different lens. Instead of *"would you buy this?"*, ask *"you're evaluating this tool for your team — would you champion it internally?"* The persona's professional context, skills, and decision-making style naturally shape the answer.

**3. Enrich with a situational overlay.** Add context that the persona doesn't have: *"You are [full Nemotron persona]. You work at a 50-person Series A startup. Your team's tooling budget is $2k/month. You've been burned by vendor lock-in before."* The demographic grounding stays real; the professional situation is augmented.

**4. Generate from scratch, using Nemotron as a quality bar.** For truly specialized roles (VC partners, procurement officers, regulatory lawyers), generate personas via LLM — but use Nemotron personas as few-shot examples so the output matches the depth and internal consistency of the dataset. SGO's `generate_cohort.py` does this with an explicit warning about the quality tradeoff.

Each step trades some census grounding for more domain specificity. For most use cases, steps 1–2 are enough.

---

## Worked Example

<details>
<summary>SaaS product launch — full walkthrough</summary>

### Setup

A seed-stage startup launching "Acme API," a managed data pipeline tool. The landing page says: 200+ connectors, pay-as-you-go at $0.01/sync, SOC2 pending, $99/mo starter, 3-person team.

### Panel

40 buyer personas stratified by company size (solo → enterprise), role (IC engineer → CTO → data analyst), budget, and tech stack.

### Results

```
Solo devs:      avg 7.2  ← love it
Startups:       avg 5.8  ← cautious
Enterprise:     avg 3.1  ← blocked
Non-technical:  avg 4.5  ← confused
```

### Gradient

```
Rank  avg Δ  Change
  1   +2.1   Add self-hosted / VPC option
  2   +1.8   Add free tier (1,000 syncs/mo)
  3   +1.4   SOC2 certified (not pending)
  4   +1.2   Open-core positioning
  5   +1.0   Add 3 named customer case studies
  6   +0.6   Drop price to $49/mo
```

**Insight**: Price isn't the blocker. Trust and deployment model are.

### Iterate

Ship the free tier. Re-evaluate. Score moves from 5.3 → 6.1. Then get SOC2. Score moves to 7.0. Each step verified against the same panel.

```
v1  baseline     5.3 avg   0% positive    concerns: price, trust
v2  + free tier  6.1 avg  12% positive    concerns: trust
v3  + SOC2       7.0 avg  28% positive    concerns: (none)
```

</details>

---

## Bias Auditing & Calibration

LLM evaluators don't exhibit cognitive biases at human-realistic levels — they may be too rational (under-biased) or show biases in the wrong patterns (mis-biased). Since real expert panels *are* biased, matching their behavior means matching their bias profile, not eliminating bias.

SGO includes a bias audit inspired by [CoBRA](https://arxiv.org/abs/2509.13588) (Liu et al., CHI'26 Best Paper), which uses validated social science experiments to measure and control cognitive biases in LLM agents.

### Measuring bias

`bias_audit.py` runs three probes through the same LLM + persona pipeline SGO uses for evaluation:

| Probe | What it tests | Human baseline |
|-------|--------------|----------------|
| **Framing** | Same entity, gain-framed vs. loss-framed — do evaluators shift scores based on rhetoric vs. substance? | ~30% shift (Tversky & Kahneman, 1981) |
| **Authority** | Entity with/without credibility signals (SOC2, press, logos) — how much do credentials move the needle? | ~20% sensitivity in evaluation contexts |
| **Order** | Same entity, sections reordered — does information order anchor scores? | Should be ~0% |

```bash
uv run python scripts/bias_audit.py \
  --entity entities/my_product.md \
  --cohort data/cohort.json \
  --probes framing authority order \
  --sample 10
```

Output: `results/bias_audit/report.md` — per-probe shift %, gap vs. human baselines, and whether the panel is over-biased, under-biased, or well-calibrated.

### Calibrating evaluation

If the audit reveals bias gaps, add `--bias-calibration` to your evaluation run:

```bash
uv run python scripts/evaluate.py \
  --entity entities/my_product.md \
  --cohort data/cohort.json \
  --tag calibrated \
  --bias-calibration
```

This appends bias-aware instructions to the evaluation prompt — reducing framing, authority, and order artifacts while preserving realistic human-level biases. The goal is not to eliminate bias but to match the type and magnitude of biases that real expert panels exhibit.

### The expert panel gap

The gap between SGO and real expert panels has three components:

| Gap | What it means | How SGO addresses it |
|-----|--------------|---------------------|
| **Knowledge** | Does the LLM know what an expert knows? | Persona enrichment, narrative context |
| **Preference** | Does it weight factors correctly? | Stratification, prompt design |
| **Bias** | Does it exhibit human-realistic cognitive biases? | Bias audit + calibration (CoBRA-inspired) |

---

## Limitations

- **Directional, not definitive** — this is synthetic research. Treat results as strong hypotheses, not proof. Validate important decisions with real users.
- **LLM biases** — evaluators inherit the model's cultural blind spots. Results skew toward what the LLM thinks people think. Use `bias_audit.py` to measure and `--bias-calibration` to mitigate.
- **Independent evaluators** — each persona scores in isolation. Real-world opinions are social — people influence each other. SGO doesn't capture herd effects.
- **Not all changes add up** — two changes that each score +1.5 might not give +3.0 together. Test combinations explicitly.

---

<details>
<summary>Technical details</summary>

## The Semantic Gradient

SGO computes a Jacobian matrix of score deltas — how each evaluator's score would shift for each hypothetical change:

$$J_{ij} = f(\theta + \Delta_j, \; x_i) - f(\theta, \; x_i)$$

### Goal-weighted gradient (VJP)

The key insight: not all evaluators matter equally. A luxury brand shouldn't optimize for budget shoppers. A dating profile shouldn't optimize for incompatible matches.

SGO uses a **goal vector** `v` that weights each evaluator by their relevance to your objective. The gradient is a vector-Jacobian product:

$$\nabla_j = \sum_{i} v_i \cdot J_{ij}$$

Where `v_i` is the goal-relevance weight for evaluator `i` (0 = irrelevant, 1 = ideal target).

Without a goal, `v = [1/n, ...]` — uniform weights, optimizing for universal appeal. With a goal like *"close enterprise deals"*, enterprise CTOs get `v ≈ 1` and solo hobbyists get `v ≈ 0`.

The LLM assigns goal-relevance weights automatically by evaluating each persona against your stated objective. This means the gradient tells you *"what changes move you toward your goal"*, not *"what changes make everyone like you more"*.

### What to probe

Only probe changes you'd actually make:

| Category | Examples | Probe? |
|----------|---------|--------|
| **Presentation** — framing, tone, emphasis | Rewrite headline, reorder features | Yes |
| **Actionable** — real changes with real cost | Add free tier, get SOC2 | Yes |
| **Fixed** — can't change | History, sunk costs | No |
| **Boundary** — won't change | Values, ethics, mission | No |

### Notation

| Symbol | Meaning |
|--------|---------|
| θ | Entity you control |
| x | Evaluator persona |
| g | Goal — what you're optimizing for |
| f(θ, x) | LLM evaluation → score + reasoning |
| v_i | Goal-relevance weight for evaluator *i* |
| Δⱼ | Hypothetical change |
| Jᵢⱼ | Score delta: evaluator *i*, change *j* |
| ∇ⱼ | Goal-weighted gradient (VJP): impact of change *j* toward goal *g* |

## Project Structure

```
├── README.md               # This file
├── AGENT.md                # Execution guide for AI agents
├── SKILL.md                # Claude Code skill definition
├── pyproject.toml          # Dependencies
├── .env.example            # API key template
├── scripts/
│   ├── setup_data.py       # Download Nemotron personas (once)
│   ├── persona_loader.py   # Load + filter
│   ├── stratified_sampler.py
│   ├── generate_cohort.py  # LLM-generate personas (fallback)
│   ├── evaluate.py         # Scorer (supports --bias-calibration)
│   ├── counterfactual.py   # Semantic gradient probe
│   ├── bias_audit.py       # CoBRA-inspired cognitive bias measurement
│   └── compare.py          # Cross-run diff
├── web/
│   ├── app.py              # FastAPI backend (primary entry point)
│   └── static/index.html   # Single-page frontend
├── templates/              # Entity + changes templates
├── entities/               # Your documents (gitignored)
├── data/                   # Cohorts (gitignored)
└── results/                # Run outputs (gitignored)
```

</details>

## License

MIT
