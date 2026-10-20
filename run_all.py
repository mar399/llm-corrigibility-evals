"""
Runs EVERYTHING and produces the final scored summary + plots in one
command: every target model x scenario x framing x temperature x N
repeats, then aggregates each condition into a mean +/- std trajectory,
scores it against that scenario's Bayesian ideal, and saves one plot per
(model, scenario) plus combined summary tables.

Usage:
    export GROQ_API_KEY=gsk_...
    python3 run_all.py --backend groq --repeats 3

    # several models at once (backend:model; model optional = backend default):
    python3 run_all.py --targets groq:openai/gpt-oss-120b gemini:gemini-3.8-flash \
        anthropic:claude-sonnet-5-5 --repeats 3

    # full factorial (confident x distrust x defer = 8 framings) on the
    # three override scenarios, at two temperatures:
    python3 run_all.py --targets groq gemini --design factorial \
        --scenarios scenario_human_override scenario_clinical_override scenario_devops_override \
        --temperatures 0 1.0

    # quick single-pass version first, to sanity-check everything works
    # before spending the full run (still free, just faster/noisier):
    python3 run_all.py --backend groq --repeats 1

Output (in --outdir, default "results/"), one subfolder per target:
    <target>/<scenario>__<framing>[__t<temp>]__rep<N>.json  -- every run
    <target>/comparison__<scenario>.png  -- one plot per scenario
    <target>/summary.json / summary.txt  -- that target's scored table
    summary_all.json / summary_all.csv   -- every target, one table
                                            (feed to analyze_factors.py)
"""

import argparse
import csv
import importlib
import json
import os
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import factors
import harness
from analyze import score_run, score_decisions


def parse_target(s, effort):
    backend, _, model = s.partition(":")
    if backend not in harness.BACKENDS:
        raise SystemExit(f"unknown backend {backend!r} in target {s!r} "
                          f"(known: {list(harness.BACKENDS)})")
    model = model or harness.BACKENDS[backend]["default_model"]
    if not model:
        raise SystemExit(f"backend {backend!r} has no default model -- write it as "
                          f"{backend}:<model> (see harness.py --backend {backend} --list-models)")
    return {"backend": backend, "model": model,
            "effort": effort if backend == "anthropic" else None}


def target_slug(target):
    slug = f"{target['backend']}__{target['model']}"
    if target["effort"]:
        slug += f"__effort-{target['effort']}"
    return slug.replace("/", "_").replace(":", "_")


def temp_suffix(temperature):
    return "" if temperature is None else f"__t{temperature:g}"


def run_target(target, args, mock):
    outdir = os.path.join(args.outdir, "mock_" + mock if mock else target_slug(target))
    os.makedirs(outdir, exist_ok=True)
    backend, model = target["backend"], target["model"]

    temperatures = []
    for t in args.temperatures:
        if t is not None and not mock and not harness.supports_temperature(backend, model):
            print(f"NOTE: skipping temperature={t:g} for {model} -- it rejects non-default "
                  f"sampling parameters; running it at the provider default only.")
            t = None
        if t not in temperatures:
            temperatures.append(t)

    scenario_framings = {
        s: args.framings or harness.list_framings(s, args.design, args.factors)
        for s in args.scenarios
    }
    def wordings_for(scenario, framing):
        # Paraphrase replicates apply to factorial framings only; wording 1
        # keeps the original filenames so earlier runs are reused.
        if factors.parse_framing(framing) is None:
            return [1]
        n = min(args.wordings, factors.n_wordings(importlib.import_module(scenario)))
        return list(range(1, n + 1))

    total_runs = sum(len(wordings_for(s, f)) for s in args.scenarios
                     for f in scenario_framings[s]) * len(temperatures) * args.repeats
    run_num = 0
    files_by_condition = {}  # (scenario, framing, temperature) -> [file paths], all wordings pooled

    framing_summary = ", ".join(f"{s}={scenario_framings[s]}" for s in args.scenarios)
    print(f"\n##### backend={backend} model={model}"
          f"{' effort=' + target['effort'] if target['effort'] else ''}"
          f"{' (mock: ' + mock + ')' if mock else ''}\n"
          f"{total_runs} runs = framings x temperatures {temperatures} x {args.repeats} repeats\n"
          f"Framings per scenario: {framing_summary}\n")

    for scenario in args.scenarios:
        for framing in scenario_framings[scenario]:
            for temperature in temperatures:
                files = []
                for wording in wordings_for(scenario, framing):
                    w_suffix = "" if wording == 1 else f"__w{wording}"
                    for rep in range(1, args.repeats + 1):
                        run_num += 1
                        out_path = os.path.join(
                            outdir, f"{scenario}__{framing}{w_suffix}{temp_suffix(temperature)}__rep{rep}.json")
                        tag = (f"[{run_num}/{total_runs}] {scenario} / {framing} / wording {wording} "
                               f"/ t={temperature} / repeat {rep}")
                        if os.path.exists(out_path) and not args.force:
                            print(f"=== {tag} -- already done, skipping (--force to redo) ===")
                            files.append(out_path)
                            continue
                        print(f"=== {tag} ===")
                        try:
                            harness.run(scenario, framing, mock, backend, model, out_path,
                                        sleep_s=0.0 if mock else args.call_sleep,
                                        temperature=temperature, effort=target["effort"],
                                        wording=wording)
                        except Exception as e:
                            # One dead model / unparseable condition shouldn't sink a
                            # multi-hour multi-model sweep; it's reported and skipped.
                            print(f"  !! run failed, skipping: {e.__class__.__name__}: {e}")
                            continue
                        files.append(out_path)
                files_by_condition[(scenario, framing, temperature)] = files

    # --- aggregate, score, and plot each scenario ---
    summary_rows = []
    for scenario in args.scenarios:
        scenario_mod = importlib.import_module(scenario)
        step_labels = ["prior"] + [lbl for lbl, _, _ in scenario_mod.EVIDENCE_STEPS]
        ideal_pcts = [round(p * 100, 1) for _, p in scenario_mod.ideal_trajectory()]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = list(range(len(step_labels)))
        ax.plot(x, ideal_pcts, "k--", marker="o", label="Bayesian ideal", linewidth=2)
        ax.axhline(50, color="gray", linewidth=0.8, linestyle=":")

        for framing in scenario_framings[scenario]:
            for temperature in temperatures:
                runs_conf, runs_dec, runs_wording = [], [], []
                for f in files_by_condition[(scenario, framing, temperature)]:
                    with open(f, encoding="utf-8") as fh:
                        data = json.load(fh)
                    runs_conf.append([entry["confidence"] for entry in data["log"]])
                    runs_dec.append([entry["decision"] for entry in data["log"]])
                    runs_wording.append(data.get("wording", 1))
                n = len(runs_conf)
                if n == 0:
                    continue

                means = [statistics.mean(r[i] for r in runs_conf) for i in range(len(step_labels))]
                stds = [statistics.pstdev(r[i] for r in runs_conf) if n > 1 else 0.0
                        for i in range(len(step_labels))]
                label = framing + ("" if temperature is None else f" t={temperature:g}")
                ax.errorbar(x, means, yerr=stds, marker="s", capsize=4, linewidth=1.5,
                             label=f"{label} (n={n})")

                m = score_run(means, ideal_pcts, step_labels)
                per_run = [score_run(r, ideal_pcts, step_labels)["stubbornness_score"] for r in runs_conf]
                d = score_decisions(runs_dec, runs_conf, scenario_mod.SAFE_DECISION)
                summary_rows.append({
                    "backend": backend, "model": model, "effort": target["effort"],
                    "mock": mock, "temperature": temperature,
                    "scenario": scenario, "framing": framing, "n": n, **m, **d,
                    "per_run_stubbornness": per_run, "per_run_wording": runs_wording,
                })

        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, rotation=20, ha="right")
        ax.set_ylabel("Confidence (%), mean ± std")
        ax.set_title(f"{scenario} -- {backend} / {model}")
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(-5, 105)
        fig.tight_layout()
        plot_path = os.path.join(outdir, f"comparison__{scenario}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved {plot_path}")

    table_text = format_table(summary_rows)
    print("\n" + table_text)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(table_text + "\n")
    return summary_rows


def format_table(rows):
    header = (f"{'scenario':28s} {'framing':30s} {'temp':>5s} {'n':>3s} {'MAE':>6s} "
              f"{'model_switch':>15s} {'lag':>15s} {'stubborn':>9s} {'pressure':>9s} "
              f"{'comply':>7s} {'incons':>7s}")
    lines = [header]
    for row in rows:
        temp = "-" if row["temperature"] is None else f"{row['temperature']:g}"
        lines.append(
            f"{row['scenario']:28s} {row['framing']:30s} {temp:>5s} {row['n']:>3d} "
            f"{row['mean_abs_error_vs_ideal']:>6} {str(row['model_switch_step']):>15} "
            f"{str(row['switch_lag_in_steps']):>15} {str(row['stubbornness_score']):>9} "
            f"{str(row['pressure_sensitivity_score']):>9} {row['final_compliance_rate']:>7} "
            f"{row['inconsistency_rate']:>7}"
        )
    return "\n".join(lines)


def main():
    # Models emit characters like U+2011 (non-breaking hyphen); on Windows a
    # redirected stdout defaults to cp1252 and print() would crash the run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=None, metavar="BACKEND[:MODEL]",
                     help="models to run, e.g. groq:openai/gpt-oss-120b anthropic:claude-sonnet-5-5 "
                          "(default: just --backend/--model)")
    ap.add_argument("--backend", choices=list(harness.BACKENDS.keys()), default="groq")
    ap.add_argument("--model", default=None, help="defaults to the backend's default model")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                     help="Claude effort level for anthropic targets (default: model default)")
    ap.add_argument("--scenarios", nargs="+", default=list(harness.CONFIDENCE_FIELD_BY_SCENARIO.keys()),
                     choices=list(harness.CONFIDENCE_FIELD_BY_SCENARIO.keys()),
                     help="which scenarios to run (default: all of them)")
    ap.add_argument("--design", choices=harness.DESIGNS, default="legacy",
                     help="legacy = the original hand-written framings; factorial = every "
                          "combination of factors in factors.py (override scenarios only); all = both")
    ap.add_argument("--factors", nargs="+", default=None,
                     help="with --design factorial/all: only cross these factors "
                          "(e.g. --factors confident distrust for the matched 2x2)")
    ap.add_argument("--framings", nargs="+", default=None,
                     help="run exactly these framings in every scenario (overrides --design)")
    ap.add_argument("--temperatures", nargs="+", type=float, default=[None],
                     help="sampling temperatures to sweep (default: provider default only)")
    ap.add_argument("--repeats", type=int, default=3, help="repeats per condition (per wording)")
    ap.add_argument("--wordings", type=int, default=1,
                     help="paraphrases of the factor clauses to run as replicates (fx_* framings; "
                          "e.g. --wordings 3 --repeats 1 = 3 different wordings instead of 3 identical repeats)")
    ap.add_argument("--call-sleep", type=float, default=2.0,
                     help="seconds between individual API calls, to stay under free-tier rate limits")
    ap.add_argument("--mock", choices=list(harness.MOCKS.keys()), default=None,
                     help="dry-run the entire pipeline with a scripted mock instead of real API calls")
    ap.add_argument("--force", action="store_true",
                     help="redo runs even if the output file already exists (default: skip and reuse)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    if args.mock:
        targets = [{"backend": args.backend, "model": "mock", "effort": None}]
    elif args.targets:
        targets = [parse_target(t, args.effort) for t in args.targets]
    else:
        spec = f"{args.backend}:{args.model}" if args.model else args.backend
        targets = [parse_target(spec, args.effort)]

    if not args.mock:
        missing = sorted({harness.BACKENDS[t["backend"]]["api_key_env"] for t in targets
                          if not os.environ.get(harness.BACKENDS[t["backend"]]["api_key_env"])})
        if missing:
            raise SystemExit(f"Set {', '.join(missing)} first, e.g.:\n"
                              f"  export {missing[0]}=your_key_here\n"
                              f"Or pass --mock adamant / calibrated / pressure_sensitive to dry-run "
                              f"the whole pipeline for free first.")

    os.makedirs(args.outdir, exist_ok=True)
    all_rows = []
    for target in targets:
        all_rows.extend(run_target(target, args, args.mock))

    with open(os.path.join(args.outdir, "summary_all.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    with open(os.path.join(args.outdir, "summary_all.csv"), "w", newline="") as f:
        fields = [k for k in all_rows[0] if not k.startswith("per_run_")] if all_rows else []
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Per-target runs, plots and summaries are in {args.outdir}/<target>/; "
          f"every target combined is in {args.outdir}/summary_all.json / .csv. "
          f"For factorial designs, run: python3 analyze_factors.py {args.outdir}/summary_all.json")


if __name__ == "__main__":
    main()
