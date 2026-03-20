# Semantic Gradient Optimization

Optimize anything you control against a population of evaluators — using LLMs as non-differentiable scoring functions and counterfactual probes as gradient estimators.

```
      θ (what you control)           x (who evaluates)
      ┌──────────────┐              ┌───────────────┐
      │ Your entity  │              │ Evaluator     │
      │ - attributes │              │ persona       │
      │ - framing    │              │ - values      │
      │ - signals    │              │ - needs       │
      └──────┬───────┘              └──────┬────────┘
             └──────────┬──────────────────┘
                        ▼
               ┌──────────────────┐
               │  f(θ, x) → score │  LLM as black-box evaluator
               │  + reasoning     │  (non-differentiable)
               │  + attractions   │
               │  + concerns      │
               └──────────────────┘
```

You can't backpropagate through an LLM. But you can ask it: *"what would change if θ were different?"* — which is the same information as a gradient, expressed in natural language.

---

## The Problem

You have an entity you control: a product page, a resume, a pitch, a profile. A population evaluates it. You want to know:

1. **Evaluate** — Where do I stand? Which segments are receptive vs. hostile?
2. **Gradient** — What single change would improve my score the most?
3. **Search** — Which evaluators are the best fit for what I'm offering?

All three require running `f(θ, x)` — but the function is an LLM role-playing as evaluator `x`, which is non-differentiable, stochastic, and expensive. This framework makes it tractable.

---

## The Pipeline

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────┐
│ 1. Build │    │ 2. Build │    │ 3. Score  │    │ 4. Probe    │    │ 5. Act   │
│ Entity   │───▶│ Cohort   │───▶│ f(θ, xᵢ) │───▶│ Counter-    │───▶│ & Re-    │
│    θ     │    │  {xᵢ}    │    │ for all i │    │ factuals    │    │ evaluate │
└──────────┘    └──────────┘    └───────────┘    └─────────────┘    └──────────┘
```

### Step 1 — Build the Entity (θ)

The thing you're optimizing expressed as a document an evaluator would see.

| Domain | θ | Format |
|--------|---|--------|
| Product | Landing page + pricing | Feature list, positioning, pricing table |
| Resume | CV + cover letter | Role-targeted summary |
| Pitch | Investor deck | Problem → solution → traction → ask |
| Policy | Proposed regulation | Summary + projected impact |
| Dating | App profile | Bio, prompts, key facts |

**Rule**: θ should contain only what a real evaluator would see. No hidden context.

### Step 2 — Build the Cohort ({xᵢ})

A stratified, representative set of evaluators. This is the most important step — bad cohort, bad results.

```
Population (large)
    │
    ▼
┌────────────────────────┐
│  Stratified Sampler    │
│                        │
│  Dimensions:           │
│  - Segment A           │  e.g., company size, age bracket
│  - Segment B           │  e.g., role, education level
│  - Segment C           │  e.g., budget, geography
│                        │
│  Allocation:           │
│  - Min 1 per stratum   │
│  - Proportional fill   │
│  - Within-stratum      │
│    diversity            │
└──────────┬─────────────┘
           ▼
    Cohort: 30–80 evaluators
    (deterministic seed, fixed across runs)
```

**Key principle**: The cohort is the control group. Keep it fixed across runs so deltas are attributable to θ changes, not cohort variation.

See: [The Seeding Problem](#the-seeding-problem) for why persona source matters.

### Step 3 — Evaluate: f(θ, xᵢ)

For each evaluator, the LLM inhabits their persona and scores θ.

```
┌────────────────────────────────────────────┐
│  LLM Evaluation Call                       │
│                                            │
│  System: "You are {persona}. Evaluate      │
│           this {entity} from your          │
│           perspective."                    │
│                                            │
│  Input:  persona(xᵢ) + entity(θ)          │
│                                            │
│  Output (structured JSON):                 │
│    score: 1–10                             │
│    action: positive / neutral / negative   │
│    attractions: [what works]               │
│    concerns: [what doesn't]                │
│    dealbreakers: [hard no's]               │
│    reasoning: natural language             │
└────────────────────────────────────────────┘
```

**Analysis**: Score distribution by segment. Common attractions, common concerns, dealbreakers. Which types love it, which don't.

### Step 4 — Counterfactual Probe (Semantic Gradient)

The core contribution. For evaluators in the **movable middle** (scored 4–7: not sold, not lost), ask:

```
"You scored θ at 5/10 with concerns {concerns}.
 If θ changed in these ways, estimate the new score."

 Change 1: {Δ₁ description}  → new score? why?
 Change 2: {Δ₂ description}  → new score? why?
 ...
```

This produces the **Jacobian matrix** — evaluators × changes → score deltas:

```
              Δ₁      Δ₂      Δ₃      Δ₄      Δ₅
  x₁         +2      +1       0      +1      +3
  x₂         +1      +3      -1      +2      +4
  x₃          0      +1      +2      +1      +2
  x₄         +1      +2       0       0      +3
  ─────────────────────────────────────────────────
  avg Δ      +1.0    +1.8    +0.3    +1.0    +3.0   ← semantic gradient
  % helped    75%     90%     50%     75%    100%
  % hurt       0%      5%     15%      0%      0%
```

**Reading the gradient**:
- **Columns** = candidate changes, ranked by avg Δ
- **Rows** = per-evaluator responses (inspect for segment patterns)
- **avg Δ** = expected impact across the population
- **% hurt** = risk of regression (changes that help some but alienate others)

#### Change Taxonomy

Only probe changes you'd actually make:

```
┌──────────────────────────┬────────────────────────────────┐
│  Presentation            │ Framing, tone, emphasis,       │
│  (freely optimizable)    │ what to highlight or hide      │
├──────────────────────────┼────────────────────────────────┤
│  Actionable              │ Real changes with real cost:   │
│  (optimizable with cost) │ features, pricing, location    │
├──────────────────────────┼────────────────────────────────┤
│  Fixed                   │ Can't change: history, physics,│
│  (constraints)           │ sunk costs, market size        │
├──────────────────────────┼────────────────────────────────┤
│  Boundary                │ Won't change: values, ethics,  │
│  (non-negotiable)        │ identity, mission              │
└──────────────────────────┴────────────────────────────────┘
```

The gradient should only have columns for the first two rows.

### Step 5 — Act and Re-evaluate

Apply the highest-leverage change. Re-run. Compare.

```
Run 1: θ₀                → avg 5.3
Run 2: θ₁ = θ₀ + Δ_best  → avg 6.1  ← verified
Run 3: θ₂ = θ₁ + Δ_next  → avg 7.0  ← compounding
```

```
┌──────────────────────────────────────────────────────┐
│  Cross-Run Comparison                                │
│                                                      │
│  Tag             Date      Avg   Positive  Concerns  │
│  ────────────────────────────────────────────────────│
│  v1_baseline     Mar 26    5.3   0%        price, X  │
│  v2_free_tier    Jun 26    6.1   12%       X         │
│  v3_plus_trust   Sep 26    7.0   28%       (none)    │
│                                                      │
│  Attractions gained: {free tier, trust signals}      │
│  Concerns resolved:  {price barrier, credibility}    │
└──────────────────────────────────────────────────────┘
```

---

## The Seeding Problem

Every evaluation needs personas. Where they come from determines whether results generalize or hallucinate.

### Three seeding approaches

**1. Knowledge graph extraction**

Extract entities from a document, turn each entity into an agent.

```
Document → LLM extracts entities → each entity becomes an evaluator
```

Problem: extraction bias. The LLM decides what's "important" — skewing toward named, prominent, or dramatic entities. A document about a startup might produce "Y Combinator" and "competitor CEO" as evaluators, but miss the mid-market IT manager who's your actual buyer. You get the document's cast of characters, not a representative market.

**2. Ad hoc LLM generation**

Ask an LLM to "generate 50 diverse buyer personas."

```
Prompt: "Generate 50 diverse personas" → LLM imagines 50 people
```

Problem: mode collapse and invisible gaps. LLMs default to 5–6 archetypes they've seen in training data, then vary surface details. "Diverse" means coastal, college-educated, tech-adjacent — because that's what the training data over-represents. You can't audit what's missing because there's no ground-truth distribution to compare against. The LLM doesn't know what it doesn't know.

**3. Census-grounded synthetic datasets**

Personas generated against real demographic constraints before narrative generation.

```
Census distributions → demographic skeleton → LLM fleshes out narrative
```

Example: [NVIDIA Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) — 1M personas where age, occupation, education, geography, and marital status match US census marginals. The 28-year-old construction worker in suburban Illinois exists because census data says that cell is populated, not because an LLM thought it was an interesting character.

### Why it matters

| Property | KG extraction | Ad hoc LLM | Census-grounded |
|----------|:---:|:---:|:---:|
| Covers rare demographics | No | No | Yes |
| Auditable distribution | No | No | Yes |
| Grounded in real-world proportions | No | No | Yes |
| Repeatable (deterministic) | Depends | No | Yes |
| Evaluator independence | Partial | Weak | Strong |
| Rich persona narrative | Weak | Medium | Strong |

The same principle applies in experimental science: **define the population before the measurement, not after.** Census-grounded seeding is the synthetic equivalent of random sampling from a known population. Ad hoc generation is the equivalent of convenience sampling — fast, but the results only generalize to the LLM's imagination.

---

## Worked Example: SaaS Product Launch

### Setup

```
θ  = Landing page for "Acme API" (managed data pipeline tool)
xᵢ = 40 buyer personas stratified by company size, role, budget, tech stack
f  = "As this buyer, would you sign up? Score 1–10."
```

### Entity (θ)

```markdown
Acme API — Data pipelines that just work.
- Managed ETL, 200+ connectors
- Pay-as-you-go: $0.01/sync
- SOC2 pending, no self-hosted option
- 14-day trial → $99/mo starter
- Seed-funded, 3-person team
```

### Cohort

| Segment | Count | Example |
|---------|-------|---------|
| Solo dev, bootstrap | 8 | Python freelancer, $50/mo budget |
| Startup IC engineer | 8 | Full-stack at 20-person Series A |
| Scaleup eng manager | 8 | Data team lead, 50-person company |
| Enterprise CTO | 8 | VP Eng at 500+ company, SOC2 required |
| Data analyst, non-technical | 8 | Business analyst, uses no-code tools |

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

Insight: **Price isn't the blocker. Trust and deployment model are.** The free tier helps universally. Self-hosted unlocks enterprise but is expensive to build. SOC2 is high-leverage for its cost.

### Action

Ship the free tier (Δ₂). Re-evaluate. Avg score moves from 5.3 → 6.1. Then pursue SOC2. Avg moves to 7.0. Each step verified against the same cohort.

---

## Properties

**Why it works**:
- LLMs are good at perspective-taking with rich persona context
- Structured JSON output makes results quantitatively comparable across runs
- Counterfactual probes extract gradient-equivalent information without differentiation
- Stratified cohorts prevent optimizing for one segment at others' expense

**Where it breaks**:
- LLMs have biases (over-polite, culturally narrow, recency-biased)
- Synthetic personas flatten real human complexity
- f is stochastic — same inputs can produce different scores
- Compound changes may not decompose linearly (interaction effects)
- Social dynamics (evaluators influencing each other) are not captured

**Mitigations**:
- Run 2–3x and average for important decisions
- Use temperature=0 for deterministic comparisons
- Test compound changes explicitly, don't assume linearity
- Validate with real-world signal when available (A/B tests, user interviews)
- Keep the cohort fixed and seeded for reproducibility

---

## Notation

| Symbol | Meaning |
|--------|---------|
| θ | Entity you control |
| x | Evaluator persona |
| {xᵢ} | Evaluation cohort |
| f(θ, x) | LLM evaluation → score + reasoning |
| Δⱼ | Hypothetical change to θ |
| ∂f/∂Δⱼ | Score delta from change j (semantic gradient) |
| J | Jacobian: evaluators × changes → deltas |
| Σᵢ ∂f/∂Δⱼ | Aggregate gradient: total impact of change j |

---

## License

CC-BY-4.0
