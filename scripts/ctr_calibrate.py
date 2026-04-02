"""
CTR calibration — maps SGO evaluation scores to click-through rate predictions.

Given a set of "anchor" SGO runs with known real-world CTRs, fits a calibration
function (Platt scaling) that converts SGO output distributions into CTR estimates.

Usage:
    # 1. Create anchors file with known CTRs
    cat > data/ctr_anchors.json << 'EOF'
    [
        {"tag": "ad_a", "real_ctr": 0.012},
        {"tag": "ad_b", "real_ctr": 0.038},
        {"tag": "ad_c", "real_ctr": 0.006}
    ]
    EOF

    # 2. Fit calibration and predict for a new run
    uv run python scripts/ctr_calibrate.py \
      --anchors data/ctr_anchors.json \
      --predict-tag new_ad

    # 3. Convert counterfactual deltas to CTR deltas
    uv run python scripts/ctr_calibrate.py \
      --anchors data/ctr_anchors.json \
      --predict-tag new_ad \
      --with-gradient
"""

import json
import math
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_sgo_features(tag):
    """Extract prediction features from an SGO evaluation run."""
    run_dir = PROJECT_ROOT / "results" / tag
    with open(run_dir / "raw_results.json") as f:
        results = json.load(f)

    valid = [r for r in results if "score" in r]
    if not valid:
        raise ValueError(f"No valid results in {tag}")

    scores = [r["score"] for r in valid]
    actions = [r.get("action", "neutral") for r in valid]
    n = len(valid)

    return {
        "tag": tag,
        "mean_score": sum(scores) / n,
        "positive_rate": sum(1 for a in actions if a == "positive") / n,
        "champion_rate": sum(1 for s in scores if s >= 8) / n,
        "n": n,
    }


def sigmoid(x):
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def fit_platt_scaling(anchors_with_features):
    """Fit P(click) = sigmoid(a * mean_score + b) via Newton's method.

    Two parameters, tiny dataset — Newton's method with analytic gradient and
    Hessian converges in ~5-10 iterations. Minimizes MSE between sigmoid output
    and observed CTR.
    """
    xs = [a["mean_score"] for a in anchors_with_features]
    ys = [a["real_ctr"] for a in anchors_with_features]
    n = len(xs)

    a, b = 0.0, 0.0
    eps = 1e-10

    for iteration in range(50):
        # Compute gradient and Hessian of MSE loss
        g_a, g_b = 0.0, 0.0
        h_aa, h_ab, h_bb = 0.0, 0.0, 0.0

        for x, y in zip(xs, ys):
            p = sigmoid(a * x + b)
            p = max(eps, min(1 - eps, p))
            dp = p * (1 - p)          # sigmoid derivative
            ddp = dp * (1 - 2 * p)    # sigmoid second derivative
            err = p - y

            # Gradient: d/da MSE = 2/n * err * dp * x
            g_a += err * dp * x
            g_b += err * dp

            # Hessian: d²/da² MSE = 2/n * (dp² * x² + err * ddp * x²), etc.
            h_aa += (dp * dp + err * ddp) * x * x
            h_ab += (dp * dp + err * ddp) * x
            h_bb += (dp * dp + err * ddp)

        g_a *= 2.0 / n
        g_b *= 2.0 / n
        h_aa *= 2.0 / n
        h_ab *= 2.0 / n
        h_bb *= 2.0 / n

        # Solve 2x2 system: H @ step = -g
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < eps:
            break  # Hessian singular — already at optimum or degenerate

        da = -(h_bb * g_a - h_ab * g_b) / det
        db = -(h_aa * g_b - h_ab * g_a) / det

        a += da
        b += db

        if abs(da) < eps and abs(db) < eps:
            break

    return a, b


def predict_ctr(a, b, mean_score):
    return sigmoid(a * mean_score + b)


def ctr_derivative(a, b, mean_score):
    """dCTR/d(score) — used to convert score deltas to CTR deltas."""
    p = sigmoid(a * mean_score + b)
    return a * p * (1 - p)


def main():
    parser = argparse.ArgumentParser(description="CTR calibration for SGO")
    parser.add_argument("--anchors", required=True,
                        help="JSON file: [{tag, real_ctr}, ...]")
    parser.add_argument("--predict-tag", default=None,
                        help="SGO run tag to predict CTR for")
    parser.add_argument("--with-gradient", action="store_true",
                        help="Convert counterfactual deltas to CTR deltas")
    args = parser.parse_args()

    with open(args.anchors) as f:
        anchors = json.load(f)

    print(f"Loading {len(anchors)} anchor runs...\n")

    # Extract features from each anchor run
    anchors_with_features = []
    for anchor in anchors:
        try:
            features = extract_sgo_features(anchor["tag"])
            features["real_ctr"] = anchor["real_ctr"]
            anchors_with_features.append(features)
            print(f"  {anchor['tag']:20s}  real CTR: {anchor['real_ctr']:.1%}  "
                  f"SGO mean: {features['mean_score']:.1f}  "
                  f"positive: {features['positive_rate']:.0%}")
        except Exception as e:
            print(f"  {anchor['tag']:20s}  SKIP: {e}")

    if len(anchors_with_features) < 2:
        print("\nNeed at least 2 valid anchors to fit calibration.")
        return

    # Fit calibration
    a, b = fit_platt_scaling(anchors_with_features)
    print(f"\nCalibration: P(click) = sigmoid({a:.4f} * score + {b:.4f})")

    # Show calibration quality
    print("\nCalibration fit:")
    for af in anchors_with_features:
        pred = predict_ctr(a, b, af["mean_score"])
        print(f"  {af['tag']:20s}  real: {af['real_ctr']:.2%}  predicted: {pred:.2%}")

    # Predict for new tag
    if args.predict_tag:
        print(f"\n--- Prediction for '{args.predict_tag}' ---\n")
        features = extract_sgo_features(args.predict_tag)
        pred_ctr = predict_ctr(a, b, features["mean_score"])
        print(f"  SGO mean score:   {features['mean_score']:.1f}")
        print(f"  SGO positive %:   {features['positive_rate']:.0%}")
        print(f"  Predicted CTR:    {pred_ctr:.2%}")

        # Convert gradient deltas if available
        if args.with_gradient:
            cf_dir = PROJECT_ROOT / "results" / args.predict_tag / "counterfactual"
            probes_path = cf_dir / "raw_probes.json"
            if probes_path.exists():
                with open(probes_path) as f:
                    probes = json.load(f)

                deriv = ctr_derivative(a, b, features["mean_score"])
                print(f"\n  dCTR/dScore:      {deriv:.4f}")
                print(f"\n  Counterfactual CTR impact:")

                # Aggregate deltas per change
                from collections import defaultdict
                change_deltas = defaultdict(list)
                for probe in probes:
                    if not probe or "counterfactuals" not in probe:
                        continue
                    for cf in probe["counterfactuals"]:
                        change_deltas[cf.get("change_id", "?")].append(cf.get("delta", 0))

                ranked = []
                for cid, deltas in change_deltas.items():
                    avg_delta = sum(deltas) / len(deltas)
                    ctr_delta = avg_delta * deriv
                    ranked.append((cid, avg_delta, ctr_delta))
                ranked.sort(key=lambda x: x[2], reverse=True)

                for cid, score_delta, ctr_delta in ranked:
                    new_ctr = pred_ctr + ctr_delta
                    print(f"    {cid:30s}  score Δ: {score_delta:+.1f}  "
                          f"CTR Δ: {ctr_delta:+.2%}  "
                          f"→ {new_ctr:.2%}")
            else:
                print(f"\n  No counterfactual data at {cf_dir}")

    # Save calibration params
    out = {
        "a": a, "b": b,
        "n_anchors": len(anchors_with_features),
        "anchors": [{
            "tag": af["tag"],
            "real_ctr": af["real_ctr"],
            "predicted_ctr": predict_ctr(a, b, af["mean_score"]),
            "mean_score": af["mean_score"],
        } for af in anchors_with_features],
    }
    out_path = PROJECT_ROOT / "data" / "ctr_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nCalibration saved: {out_path}")


if __name__ == "__main__":
    main()
