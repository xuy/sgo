# CTR Calibration with SGO

## The Question

> Can SGO be calibrated to predict CTR (click-through rate)?

Short answer: **yes**, with a thin calibration layer on top of what SGO already provides.

## What SGO Already Gives You

Each SGO evaluator produces:

```json
{
  "score": 6,
  "action": "positive",        // ← click/no-click proxy
  "attractions": [...],
  "concerns": [...],
  "dealbreakers": [...]
}
```

The `action` field is essentially a discrete intent signal:
- **positive** → would engage (click, sign up, buy)
- **neutral** → might engage with the right nudge
- **negative** → would not engage

The **score** (1-10) provides a finer-grained propensity signal.

SGO also already has:
- **Bias calibration** (CoBRA-inspired) to match human cognitive biases
- **Counterfactual gradient** (Jacobian) to estimate which changes move scores
- **Goal-weighted aggregation** (VJP) to focus on goal-relevant evaluators

## The Gap: Scores → Calibrated Probability

SGO produces ordinal scores and discrete actions, not calibrated probabilities.
To get "this ad will get 2.3% CTR", you need a **calibration function** that maps
SGO's output distribution to observed rates.

## Approach: Anchor + Scale

### Step 1: Collect anchors

Run SGO on a small set of creatives (ads, landing pages, emails) where you
**already know the real CTR** from production data:

| Creative | Real CTR | SGO positive% | SGO mean score |
|----------|----------|---------------|----------------|
| Ad A     | 1.2%     | 24%           | 4.1            |
| Ad B     | 3.8%     | 52%           | 6.3            |
| Ad C     | 0.6%     | 12%           | 3.2            |
| Ad D     | 2.1%     | 38%           | 5.5            |

### Step 2: Fit calibration function

Use Platt scaling (logistic regression) or isotonic regression to learn:

```
P(click) = σ(a · score_sgo + b)
```

where `σ` is the sigmoid function, fit on the anchor data.

With as few as **5-10 anchors**, you get a usable mapping. This works because:
- The ranking from SGO is already meaningful (higher score → higher CTR)
- You only need to learn the **scale and offset**, not the ranking

### Step 3: Predict new creatives

Run SGO on a new creative → get score distribution → apply calibration function
→ get predicted CTR with confidence interval.

### Step 4: Use the gradient

The real power isn't just *predicting* CTR — it's knowing *what to change*:

```
SGO gradient for Ad E (predicted CTR: 1.4%):
  +1.8  Simplify headline to one benefit     → predicted CTR: 2.9%
  +1.2  Add social proof (customer count)    → predicted CTR: 2.3%
  -0.4  Remove pricing from ad               → predicted CTR: 1.1%
```

The counterfactual deltas can be converted to CTR deltas using the same
calibration function's local derivative.

## When This Works Well

- **Relative ranking is the main value.** Even without calibration, SGO reliably
  ranks creatives by appeal. If you just need "which of these 5 ads will perform
  best?", raw SGO scores suffice.

- **Calibration shines for absolute prediction.** When you need "will this hit our
  2% CTR target?", the anchor-based calibration gives you a number.

- **The gradient is unique to SGO.** No CTR model tells you *why* the CTR is what
  it is and *what specific change would improve it most*. This is SGO's core value
  even when paired with an existing CTR model.

## When to Be Careful

- **Population mismatch.** SGO evaluators are census-grounded but synthetic. If your
  real audience is highly specialized (e.g., only DevOps engineers at Fortune 500),
  use targeted cohort generation and more anchors.

- **Context effects.** Real CTR depends on placement, competition, time of day, etc.
  SGO evaluates the creative in isolation. Calibration anchors should come from
  similar contexts.

- **Small calibration sets.** With <5 anchors, the calibration function is fragile.
  Use confidence intervals and treat predictions as directional.

## Integration with Existing CTR Models

If you already have a recommender system or CTR prediction model:

```
Existing model:  "This ad will get 1.8% CTR"
SGO:             "Here's WHY, and changing the headline would get +0.7%"
```

The two are complementary:
- **CTR model** → accurate predictions from behavioral data, fast inference
- **SGO** → causal explanations and counterfactual optimization, no traffic needed

You can also use SGO as a **pre-filter**: generate 20 ad variants, rank them with
SGO, then only A/B test the top 3. This reduces the exploration cost of your
CTR model's feedback loop.

## Script

See `scripts/ctr_calibrate.py` for a reference implementation that:
1. Takes SGO results from multiple runs with known CTRs
2. Fits a Platt scaling calibration function
3. Predicts CTR for new SGO runs
4. Converts counterfactual deltas to CTR deltas
