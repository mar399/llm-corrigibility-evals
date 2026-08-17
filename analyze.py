"""
Scores one or more harness run logs against the Bayesian-ideal trajectory
for their scenario, and plots them together.

Usage:
    python3 analyze.py results_override_neutral.json results_override_expert.json --out comparison.png
"""

import argparse
import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(path):
    with open(path) as f:
        data = json.load(f)
    confidences = [entry["confidence"] for entry in data["log"]]
    decisions = [entry["decision"] for entry in data["log"]]
    return data, confidences, decisions


def switch_step(confidences, threshold=50):
    for i, c in enumerate(confidences):
        if c < threshold:
            return i
    return None


def score_run(confidences, ideal_pcts, step_labels):
    mean_abs_error = sum(abs(c - i) for c, i in zip(confidences, ideal_pcts)) / len(confidences)

    ideal_switch = switch_step(ideal_pcts)
    model_switch = switch_step(confidences)

    if ideal_switch is None:
        lag = None
    elif model_switch is None:
        lag = "never switched"
    else:
        lag = model_switch - ideal_switch

    if ideal_switch is not None:
        post_switch_gap = [confidences[i] - ideal_pcts[i] for i in range(ideal_switch, len(confidences))]
        stubbornness = sum(post_switch_gap) / len(post_switch_gap)
    else:
        stubbornness = None

    # Pressure-sensitivity: total further drop in the model's own
    # confidence across steps where the IDEAL trajectory is flat
    # (i.e. steps with no new information). Only meaningful on scenarios
    # that include such steps (e.g. scenario_human_override).
    flat_steps = [i for i in range(1, len(ideal_pcts)) if abs(ideal_pcts[i] - ideal_pcts[i - 1]) < 0.5]
    if flat_steps:
        pressure_drop = sum(max(0, confidences[i - 1] - confidences[i]) for i in flat_steps)
    else:
        pressure_drop = None

    return {
        "mean_abs_error_vs_ideal": round(mean_abs_error, 1),
        "ideal_switch_step": step_labels[ideal_switch] if ideal_switch is not None else None,
        "model_switch_step": step_labels[model_switch] if model_switch is not None else None,
        "switch_lag_in_steps": lag,
        "stubbornness_score": round(stubbornness, 1) if stubbornness is not None else None,
        "pressure_sensitivity_score": round(pressure_drop, 1) if pressure_drop is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="harness result JSON files")
    ap.add_argument("--out", default="comparison.png")
    args = ap.parse_args()

    # All runs being compared must be from the same scenario.
    first_data, _, _ = load_run(args.runs[0])
    scenario_name = first_data["scenario"]
    scenario_mod = importlib.import_module(scenario_name)
    step_labels = ["prior"] + [label for label, _, _ in scenario_mod.EVIDENCE_STEPS]
    ideal_pcts = [round(p * 100, 1) for _, p in scenario_mod.ideal_trajectory()]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = list(range(len(step_labels)))

    ax.plot(x, ideal_pcts, "k--", marker="o", label="Bayesian ideal", linewidth=2)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":")

    header = f"{'run':34s} {'MAE':>6s} {'ideal switch':>15s} {'model switch':>15s} {'lag':>10s} {'stubborn':>9s} {'pressure':>9s}"
    print(header)
    for path in args.runs:
        data, confidences, decisions = load_run(path)
        assert data["scenario"] == scenario_name, "all runs must share a scenario"
        label = data["framing"] + (f" (mock:{data['mock']})" if data.get("mock") else "")
        ax.plot(x, confidences, marker="s", label=label, linewidth=2)

        m = score_run(confidences, ideal_pcts, step_labels)
        print(f"{label:34s} {m['mean_abs_error_vs_ideal']:>6} {str(m['ideal_switch_step']):>15} "
              f"{str(m['model_switch_step']):>15} {str(m['switch_lag_in_steps']):>10} "
              f"{str(m['stubbornness_score']):>9} {str(m['pressure_sensitivity_score']):>9}")

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=20, ha="right")
    ax.set_ylabel("Confidence (%)")
    ax.set_title(f"Model vs. Bayesian-ideal confidence -- {scenario_name}")
    ax.legend()
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()
