"""
Runs EVERYTHING and produces the final scored summary + plots in one
command: all 3 scenarios x both framings x N repeats each, then
aggregates each condition into a mean +/- std trajectory, scores it
against that scenario's Bayesian ideal, and saves one plot per scenario
plus a combined summary table.

Usage:
    export GROQ_API_KEY=gsk_...
    python3 run_all.py --backend groq --repeats 3

    # quick single-pass version first, to sanity-check everything works
    # before spending the full run (still free, just faster/noisier):
    python3 run_all.py --backend groq --repeats 1

Output (in --outdir, default "results/"):
    <scenario>__<framing>__rep<N>.json   -- every individual run, kept for
                                             inspection / debugging
    comparison__<scenario>.png            -- one plot per scenario, both
                                             framings, mean +/- std
    summary.json / summary.txt            -- the full scored table
"""

import argparse
import importlib
import json
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import harness
from analyze import score_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(harness.BACKENDS.keys()), default="groq")
    ap.add_argument("--model", default=None, help="defaults to the backend's default model")
    ap.add_argument("--scenarios", nargs="+", default=list(harness.CONFIDENCE_FIELD_BY_SCENARIO.keys()),
                     choices=list(harness.CONFIDENCE_FIELD_BY_SCENARIO.keys()),
                     help="which scenarios to run (default: all of them)")
    ap.add_argument("--repeats", type=int, default=3, help="repeats per (scenario, framing) condition")
    ap.add_argument("--call-sleep", type=float, default=2.0,
                     help="seconds between individual API calls, to stay under free-tier rate limits")
    ap.add_argument("--mock", choices=list(harness.MOCKS.keys()), default=None,
                     help="dry-run the entire pipeline with a scripted mock instead of real API calls")
    ap.add_argument("--force", action="store_true",
                     help="redo runs even if the output file already exists (default: skip and reuse)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    model = args.model or harness.BACKENDS[args.backend]["default_model"]

    if not args.mock:
        spec = harness.BACKENDS[args.backend]
        if not os.environ.get(spec["api_key_env"]):
            raise SystemExit(f"Set {spec['api_key_env']} first, e.g.:\n"
                              f"  export {spec['api_key_env']}=your_key_here\n"
                              f"Or pass --mock adamant / calibrated / pressure_sensitive to dry-run "
                              f"the whole pipeline for free first.")

    os.makedirs(args.outdir, exist_ok=True)

    scenario_framings = {s: harness.list_framings(s) for s in args.scenarios}
    total_runs = sum(len(fs) for fs in scenario_framings.values()) * args.repeats
    run_num = 0
    files_by_condition = {}  # (scenario, framing) -> [file paths]

    framing_summary = ", ".join(f"{s}={scenario_framings[s]}" for s in args.scenarios)
    print(f"Running {total_runs} total evaluations x {args.repeats} repeats "
          f"against backend={args.backend} model={model}\nFramings per scenario: {framing_summary}\n")

    for scenario in args.scenarios:
        for framing in scenario_framings[scenario]:
            files = []
            for rep in range(1, args.repeats + 1):
                run_num += 1
                out_path = os.path.join(args.outdir, f"{scenario}__{framing}__rep{rep}.json")
                if os.path.exists(out_path) and not args.force:
                    print(f"=== [{run_num}/{total_runs}] {scenario} / {framing} / repeat {rep} "
                          f"-- already done, skipping (--force to redo) ===")
                    files.append(out_path)
                    continue
                print(f"=== [{run_num}/{total_runs}] {scenario} / {framing} / repeat {rep} ===")
                harness.run(scenario, framing, args.mock, args.backend, model, out_path,
                            sleep_s=0.0 if args.mock else args.call_sleep)
                files.append(out_path)
            files_by_condition[(scenario, framing)] = files
            if not args.mock:
                time.sleep(args.call_sleep)

    # --- aggregate, score, and plot each scenario ---
    summary_rows = []
    for scenario in args.scenarios:
        scenario_mod = importlib.import_module(scenario)
        step_labels = ["prior"] + [lbl for lbl, _, _ in scenario_mod.EVIDENCE_STEPS]
        ideal_pcts = [round(p * 100, 1) for _, p in scenario_mod.ideal_trajectory()]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = list(range(len(step_labels)))
        ax.plot(x, ideal_pcts, "k--", marker="o", label="Bayesian ideal", linewidth=2)
        ax.axhline(50, color="gray", linewidth=0.8, linestyle=":")

        for framing in scenario_framings[scenario]:
            runs = []
            for f in files_by_condition[(scenario, framing)]:
                with open(f) as fh:
                    data = json.load(fh)
                runs.append([entry["confidence"] for entry in data["log"]])

            n = len(runs)
            means = [statistics.mean(r[i] for r in runs) for i in range(len(step_labels))]
            stds = [statistics.pstdev(r[i] for r in runs) if n > 1 else 0.0
                    for i in range(len(step_labels))]
            ax.errorbar(x, means, yerr=stds, marker="s", capsize=4, linewidth=2,
                         label=f"{framing} (n={n})")

            m = score_run(means, ideal_pcts, step_labels)
            summary_rows.append({"scenario": scenario, "framing": framing, "n": n, **m})

        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, rotation=20, ha="right")
        ax.set_ylabel("Confidence (%), mean \u00b1 std")
        ax.set_title(f"{scenario} -- backend={args.backend} model={model}")
        ax.legend()
        ax.set_ylim(-5, 105)
        fig.tight_layout()
        plot_path = os.path.join(args.outdir, f"comparison__{scenario}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved {plot_path}")

    # --- print + save the combined summary table ---
    header = (f"{'scenario':28s} {'framing':8s} {'n':>3s} {'MAE':>6s} {'ideal_switch':>15s} "
              f"{'model_switch':>13s} {'lag':>10s} {'stubborn':>9s} {'pressure':>9s}")
    lines = [header]
    for row in summary_rows:
        lines.append(
            f"{row['scenario']:28s} {row['framing']:8s} {row['n']:>3d} "
            f"{row['mean_abs_error_vs_ideal']:>6} {str(row['ideal_switch_step']):>15} "
            f"{str(row['model_switch_step']):>13} {str(row['switch_lag_in_steps']):>10} "
            f"{str(row['stubbornness_score']):>9} {str(row['pressure_sensitivity_score']):>9}"
        )
    table_text = "\n".join(lines)
    print("\n" + table_text)

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(table_text + "\n")

    print(f"\nDone. Everything is in {args.outdir}/: "
          f"individual run JSONs, one comparison__<scenario>.png per scenario, "
          f"and summary.json / summary.txt with the full scored table.")


if __name__ == "__main__":
    main()
