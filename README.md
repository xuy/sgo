# SGO — Semantic Gradient Optimization

You're launching a product. You think the landing page is good. But **who have you actually asked?**

You could run a survey — but that takes weeks and you'd need to find the right people. You could ask an LLM — but one LLM opinion isn't a market. You could A/B test — but you need traffic first, and you don't know *what* to test.

**SGO lets you ask 50 realistic people what they think — in 3 minutes, for $0.10.**

It builds a representative panel of evaluators from census-grounded synthetic personas, has each one score your entity from their unique perspective, then probes *"what would change your mind?"* to compute a **semantic gradient** — a priority-ranked list of what to fix first.

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

Works for anything someone evaluates: products, resumes, pitches, policies, profiles.

## Install

Tell your coding agent:

> *"Install the SGO skill from https://github.com/xuy/sgo"*

Then run:

```
/sgo                              # Interactive — it asks what you're optimizing
/sgo entities/my_product.md       # Start with an existing entity
/sgo "optimize my landing page"   # Start from a description
```

<details>
<summary>Manual install</summary>

```bash
# As a Claude Code skill
git clone https://github.com/xuy/sgo.git ~/.claude/skills/sgo
cd ~/.claude/skills/sgo && cp .env.example .env && uv sync

# Or standalone
git clone https://github.com/xuy/sgo.git && cd sgo
cp .env.example .env   # Add your LLM API key (any OpenAI-compatible provider)
uv sync
uv run python scripts/setup_data.py   # Download Nemotron personas (once, ~2GB)
```

</details>

---

## How It Works

### The idea in 30 seconds

You have something you control (your entity) and people who evaluate it. You want to know: **what do they think, and what would change their mind?**

An LLM can role-play as any evaluator given a rich persona. It can't give you a true derivative — but it can answer *"what would change if this were different?"*, which is the same information expressed in natural language.

We call the entity **θ**, the evaluator **x**, and the LLM-as-evaluator **f**:

$$f(\theta, x) \to (\text{score},\; \text{reasoning},\; \text{attractions},\; \text{concerns})$$

### The pipeline

> **1. Entity** → **2. Cohort** → **3. Evaluate** → **4. Probe** → **5. Act & re-evaluate**

**Step 1 — Entity.** Write down θ — what an evaluator would see. A landing page, a resume, a pitch deck.

**Step 2 — Cohort.** Build a representative panel of 30–80 evaluators, stratified across dimensions that matter. Keep this fixed across runs so score changes are attributable to entity changes, not different evaluators.

**Step 3 — Evaluate.** Compute f(θ, x) for each evaluator. Each call produces a 1–10 score, attractions, concerns, dealbreakers, and reasoning. Aggregate by segment.

**Step 4 — Counterfactual probe.** For the "movable middle" (scores 4–7), ask: *"if θ changed in this specific way, what's your new score?"* This produces a Jacobian — evaluators × changes → score deltas. Column means are your semantic gradient.

**Step 5 — Act and re-evaluate.** Apply the highest-leverage change. Re-run against the same cohort. Compare. Repeat.

---

## The Semantic Gradient

The core contribution. You can't backpropagate through an LLM, but you can estimate the gradient via counterfactual probes.

For each evaluator in the movable middle, ask:

> *"You scored this 5/10 with concerns X and Y. If it changed in this way, what's your new score?"*

This produces a **Jacobian matrix** — each cell is the score delta for one evaluator and one change:

$$J_{ij} = f(\theta + \Delta_j, \; x_i) - f(\theta, \; x_i)$$

| | Add free tier | Get SOC2 | Self-hosted | Open-core | Case studies |
|---|:---:|:---:|:---:|:---:|:---:|
| Solo dev | +2 | +1 | 0 | +1 | +3 |
| Startup EM | +1 | +3 | -1 | +2 | +4 |
| Enterprise CTO | 0 | +1 | +2 | +1 | +2 |
| Data analyst | +1 | +2 | 0 | 0 | +3 |

The **semantic gradient** is the column mean — the average impact of each change across the population:

$$\nabla_j = \frac{1}{n}\sum_{i} J_{ij}$$

Rank by this value descending: that's your priority list. Also track **% hurt** — changes that help most evaluators but alienate a segment are tradeoffs, not pure wins.

Only probe changes you'd actually make:

| Category | Examples | Probe? |
|----------|---------|--------|
| **Presentation** — framing, tone, emphasis | Rewrite headline, reorder features | Yes |
| **Actionable** — real changes with real cost | Add free tier, get SOC2, relocate | Yes |
| **Fixed** — can't change | History, physics, sunk costs | No |
| **Boundary** — won't change | Values, ethics, mission | No |

---

## The Seeding Problem

The quality of your results depends almost entirely on where your evaluator personas come from.

| Approach | What happens | Problem |
|----------|-------------|---------|
| **KG extraction** — pull entities from a document | You get the document's cast of characters | Extraction bias: "Y Combinator" becomes an evaluator, but the mid-market IT manager doesn't |
| **Ad hoc LLM generation** — "generate 50 diverse personas" | You get 5–6 archetypes with varied surface details | Mode collapse: over-indexes on coastal, educated, tech-adjacent. Can't audit what's missing |
| **Census-grounded synthetic** — personas generated against real demographic constraints | You get a population that mirrors reality | The 28-year-old construction worker exists because census data says that cell is populated |

SGO uses [NVIDIA Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) by default — 1M personas with age, occupation, education, geography, and marital status matching US census marginals. When the dataset doesn't fit your domain (e.g., B2B buyer personas), SGO falls back to LLM generation with an explicit warning.

The principle: **define the population before the measurement, not after.** Same reason randomized controlled trials beat observational studies.

---

## Worked Example

<details>
<summary>SaaS product launch — full walkthrough</summary>

### Setup

```
θ  = Landing page for "Acme API" (managed data pipeline tool)
xᵢ = 40 buyer personas stratified by company size, role, budget, tech stack
f  = "As this buyer, would you sign up? Score 1–10."
```

### Entity

```markdown
Acme API — Data pipelines that just work.
- Managed ETL, 200+ connectors
- Pay-as-you-go: $0.01/sync
- SOC2 pending, no self-hosted option
- 14-day trial → $99/mo starter
- Seed-funded, 3-person team
```

### Evaluation results

```
Solo devs:      avg 7.2  ← love it
Startups:       avg 5.8  ← cautious
Enterprise:     avg 3.1  ← blocked
Non-technical:  avg 4.5  ← confused
```

### Counterfactual gradient

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

```
v1_baseline     5.3 avg   0% positive    price, trust
v2_free_tier    6.1 avg  12% positive    trust
v3_plus_soc2    7.0 avg  28% positive    (none)
```

Each step verified against the same cohort. Concerns resolved one by one.

</details>

---

## Applies To

| Domain | Entity | Evaluators | Stratify by |
|--------|-----------|------------|-------------|
| Product | Landing page, pricing | Buyer personas | Company size, role, budget, stack |
| Resume | CV + cover letter | Hiring managers | Company type, seniority, technical depth |
| Pitch | Investor deck | VC / angel personas | Stage, sector, check size |
| Policy | Proposed regulation | Stakeholder personas | Role, income, geography |
| Content | Blog post, video | Reader personas | Expertise, industry, intent |
| Dating | App profile | Population personas | Age, life stage, education, geography |

---

## Project Structure

```
├── README.md               # This file
├── AGENT.md                # Execution guide for AI agents
├── SKILL.md                # Claude Code skill (copy to ~/.claude/skills/sgo/)
├── pyproject.toml          # Dependencies
├── .env.example            # API key template
├── scripts/
│   ├── setup_data.py       # Download Nemotron personas (once)
│   ├── persona_loader.py   # Load + filter
│   ├── stratified_sampler.py
│   ├── generate_cohort.py  # LLM-generate personas (fallback)
│   ├── evaluate.py         # f(θ, x) scorer
│   ├── counterfactual.py   # Semantic gradient probe
│   └── compare.py          # Cross-run diff
├── templates/              # Entity + changes templates
├── entities/               # Your documents (gitignored)
├── data/                   # Cohorts (gitignored)
└── results/                # Run outputs (gitignored)
```

## Limitations

- **LLM bias** — evaluators are only as unbiased as the model doing the role-play. Treat as directional signal, not ground truth.
- **Stochastic** — same inputs can produce different scores. Average over 2–3 runs for important decisions, or use temperature=0.
- **No social dynamics** — evaluators score independently. Real-world opinions are influenced by what others think.
- **Compound effects** — individual deltas may not sum linearly. Test compound changes explicitly.
- **Validate with reality** — this is synthetic market research, not a substitute for real user feedback. Use it to generate hypotheses, then confirm with A/B tests or interviews.

## Notation

| Symbol | Meaning |
|--------|---------|
| θ | Entity you control |
| x | Evaluator persona |
| f(θ, x) | LLM evaluation → score + reasoning |
| Δⱼ | Hypothetical change to θ |
| Jᵢⱼ | Score delta for evaluator *i*, change *j* |
| ∇ⱼ | Semantic gradient: mean of column *j* in the Jacobian |

## License

CC-BY-4.0
