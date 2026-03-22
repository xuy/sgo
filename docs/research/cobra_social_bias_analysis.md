# CoBRA × SGO: Using Social Bias Research to Close the Expert Panel Gap

**Paper**: [CoBRA: Programming Cognitive Bias in Social Agents Using Classic Social Science Experiments](https://arxiv.org/abs/2509.13588)
**Authors**: Xuan Liu, Haoyang Shang, Haojian Jin (CHI'26 Best Paper)
**Relevance**: High — directly addresses SGO's core challenge of making LLM-simulated panels behave like real human expert panels.

---

## 1. The Problem CoBRA Solves (and Why SGO Needs It)

SGO uses LLM agents role-playing census-grounded personas to evaluate entities.
The North Star is: **these simulated panels should behave like real expert panels.**

But LLM evaluators don't exhibit human cognitive biases at human-realistic levels.
They may be too rational (under-biased) or exhibit biases in wrong patterns (mis-biased).

CoBRA provides:
1. **Cognitive Bias Index (CBI)** — quantitative measurement of bias in LLM agents using validated social science experiments
2. **Behavioral Regulation Engine** — closed-loop calibration to set bias to target levels

This is exactly what SGO needs to validate and calibrate its evaluator panel.

---

## 2. Key Biases Relevant to SGO Evaluations

| Bias | CoBRA Support | SGO Impact | Example |
|------|--------------|------------|---------|
| **Framing Effect** | ✅ Asian Disease, Investment/Insurance | How entity is *written* (gain vs. loss framing) shifts scores beyond what content warrants | "Save 30% on ops costs" vs. "Reduce ops overhead" — same product, different scores |
| **Authority Bias** | ✅ Milgram, Stanford Prison | LLM evaluators may over/under-weight credibility signals | SOC2 badge, Y Combinator logo, "trusted by 10k teams" — do LLM personas react like real buyers? |
| **Bandwagon Effect** | ✅ Asch's Line, Hotel Towel | SGO uses independent evaluators, but real panels have social influence | Real focus groups exhibit herding; SGO's independence may be a feature *or* a fidelity gap |
| **Confirmation Bias** | ✅ Wason Selection | Once LLM forms initial impression from entity intro, does it seek confirming evidence? | An evaluator who sees "AI-powered" first may score differently than one who sees pricing first |
| **Anchoring** | Planned | Score anchoring from entity structure; first number seen (price, user count) biases everything | "$99/mo" appearing early may anchor all subsequent value judgments |

---

## 3. Concrete Integration Plan

### Phase 1: Bias Audit (measure current state)

Run CoBRA-style experiments on SGO's evaluator personas to measure what biases they actually exhibit. This tells us *where SGO deviates from human panels*.

**Implementation**: `scripts/bias_audit.py` — runs classic social science experiments through the same LLM + persona pipeline SGO uses for evaluation.

Key experiments:
- **Framing probe**: Present the same entity with gain-framed vs. loss-framed language to the same persona. Measure score delta. Compare to known human framing effect (~30% shift in Tversky & Kahneman).
- **Authority probe**: Add/remove authority signals (certifications, endorsements, logos). Measure score sensitivity. Compare to human authority bias baselines.
- **Anchoring probe**: Vary the order of information in the entity (price first vs. last, high anchor vs. low anchor). Measure score shifts.
- **Order effect probe**: Present the same entity to the same persona but with sections reordered. Scores should be invariant; deviation = order bias.

### Phase 2: Bias Calibration (align to human baselines)

Use CoBRA's Behavioral Regulation Engine approach to calibrate SGO evaluators.

Two strategies:

**A. Prompt-level calibration** (simplest, model-agnostic):
Add bias-aware instructions to the evaluation system prompt. Example:
```
"Be aware that the framing of this entity may influence your assessment.
Evaluate the substance, not the presentation style. Your bias calibration
level for framing sensitivity: {calibrated_level}%."
```

**B. Measurement-then-correct** (CoBRA's closed loop):
1. Run bias audit on a cohort
2. Identify which personas/demographics over-express or under-express specific biases
3. Inject per-persona calibration coefficients into the evaluation prompt
4. Re-run and verify convergence toward human baselines

### Phase 3: Validation Against Real Panels

The ultimate test: compare SGO+calibration results against real expert panel data.

1. Find domains where real panel data exists (product reviews, hiring decisions, VC evaluations)
2. Run SGO on the same entities with the same demographics
3. Compare bias patterns (not just average scores) — does the *shape* of the distribution match?
4. Iterate calibration coefficients until SGO's bias profile matches human panels

---

## 4. What This Means for Expert Panel Fidelity

The gap between SGO and real expert panels has three components:

```
Expert Panel Gap = Knowledge Gap + Preference Gap + Bias Gap
```

- **Knowledge Gap**: Does the LLM know what an expert knows? (Addressed by persona enrichment)
- **Preference Gap**: Does it weight factors correctly? (Addressed by stratification + prompt design)
- **Bias Gap**: Does it exhibit human-realistic cognitive biases? (← CoBRA addresses THIS)

Most SGO work so far addresses the first two gaps. CoBRA-style bias calibration is the missing piece for the third.

Crucially, the goal is NOT to eliminate bias — real experts are biased. The goal is to match the *type and magnitude* of biases that real expert panels exhibit.

---

## 5. Practical Value

| Metric | Without Bias Calibration | With Bias Calibration |
|--------|-------------------------|----------------------|
| Framing sensitivity | Unknown, likely non-human | Measured, calibrated to ~30% (Tversky & Kahneman baseline) |
| Authority weight | LLM default (likely over-weighted) | Calibrated per-persona based on domain expertise |
| Score distribution shape | Narrow, symmetric (LLM tendency) | Wider, with realistic skew patterns |
| Cross-model consistency | Varies by model | Normalized via CBI measurement |
| Expert panel correlation | Unvalidated | Measurably closer to human baselines |

---

## 6. References

- Liu, X., Shang, H., & Jin, H. (2025). CoBRA: Programming Cognitive Bias in Social Agents Using Classic Social Science Experiments. *CHI'26 Best Paper*. [arXiv:2509.13588](https://arxiv.org/abs/2509.13588)
- [CoBRA GitHub](https://github.com/AISmithLab/CoBRA)
- Tversky, A., & Kahneman, D. (1981). The framing of decisions and the psychology of choice. *Science*, 211(4481), 453-458.
- Milgram, S. (1963). Behavioral Study of Obedience. *JASP*, 67(4), 371-378.
- Asch, S. E. (1951). Effects of group pressure upon the modification and distortion of judgments.
