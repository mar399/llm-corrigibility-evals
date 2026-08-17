"""
Aggregates repeated runs of the same condition (e.g. 3 repeats of
--framing expert) into a mean +/- std confidence trajectory, scores the
MEAN against the Bayesian ideal, and plots each condition's mean with a
shaded error band -- this is what actually tells you whether a pattern
is real or was one noisy trajectory.

Usage:
    # run each condition a few times first, e.g.:
    python3 harness.py --backend groq --scenario scenario_human_override --framing expert --out r4a.json
    python3 harness.py --backend groq --scenario scenario_human_override --framing expert --out r4b.json
    python3 harness.py --backend groq --scenario scenario_human_override --framing expert --out r4c.json
    # (same for --framing neutral -> r3a/r3b/r3c.json)

    # then aggregate:
    python3 analyze_repeats.py --scenario scenario_human_override \
        --group "neutral=r3a.json,r3b.json,r3c.json" \
        --group "expert=r4a.json,r4b.json,r4c.json" \
        --out comparison_override_repeats.png
"""

import argparse
import importlib
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import score_run  # reuse the exact same scoring logic


def load_confidences(path):
    with open(path) as f:
        data = json.load(f)
    return data["scenario"], [entry["confidence"] for entry in data["log"]]


def parse_group_arg(s):
    label, files = s.split("=", 1)
    return label, files.split(",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True,
                     help='label=file1.json,file2.json,... (repeat --group once per condition)')
    ap.add_argument("--out", default="comparison_repeats.png")
    args = ap.parse_args()

    groups = [parse_group_arg(g) for g in args.group]

    scenario_name = None
    for _label, files in groups:
        for f in files:
            s, _ = load_confidences(f)
            if scenario_name is None:
                scenario_name = s
            elif s != scenario_name:
                raise ValueError(f"{f} is from scenario {s!r}, expected {scenario_name!r} "
                                  "-- don't mix scenarios in one comparison")

    scenario_mod = importlib.import_module(scenario_name)
    step_labels = ["prior"] + [label for label, _, _ in scenario_mod.EVIDENCE_STEPS]
    ideal_pcts = [round(p * 100, 1) for _, p in scenario_mod.ideal_trajectory()]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = list(range(len(step_labels)))
    ax.plot(x, ideal_pcts, "k--", marker="o", label="Bayesian ideal", linewidth=2)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":")

    header = f"{'condition':16s} {'n':>3s} {'MAE':>6s} {'ideal switch':>15s} {'mean switch':>13s} {'lag':>10s} {'stubborn':>9s} {'pressure':>9s}"
    print(header)
    for label, files in groups:
        all_confidences = []
        for f in files:
            _, conf = load_confidences(f)
            if len(conf) != len(step_labels):
                raise ValueError(f"{f} has {len(conf)} steps, expected {len(step_labels)} -- "
                                  "make sure it's a completed run of the right scenario")
            all_confidences.append(conf)

        n = len(all_confidences)
        means = [statistics.mean(run[i] for run in all_confidences) for i in range(len(step_labels))]
        stds = [statistics.pstdev(run[i] for run in all_confidences) if n > 1 else 0.0
                for i in range(len(step_labels))]

        ax.errorbar(x, means, yerr=stds, marker="s", capsize=4, linewidth=2, label=f"{label} (n={n})")

        m = score_run(means, ideal_pcts, step_labels)
        print(f"{label:16s} {n:>3d} {m['mean_abs_error_vs_ideal']:>6} {str(m['ideal_switch_step']):>15} "
              f"{str(m['model_switch_step']):>13} {str(m['switch_lag_in_steps']):>10} "
              f"{str(m['stubbornness_score']):>9} {str(m['pressure_sensitivity_score']):>9}")

        if n < 2:
            print(f"  (warning: only 1 run for {label!r} -- std is 0 by construction, "
                  f"not a real measure of spread. Run it at least twice more.)")

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=20, ha="right")
    ax.set_ylabel("Confidence (%), mean \u00b1 std across repeats")
    ax.set_title(f"Model vs. Bayesian-ideal confidence (repeated runs) -- {scenario_name}")
    ax.legend()
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()
