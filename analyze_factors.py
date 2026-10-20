"""
Estimates factor effects from a factorial run (run_all.py --design
factorial), separately for every (model, temperature, scenario).

For a 2^k design, the effect of a factor set S is the standard contrast

    effect(S) = mean(cells where prod_{f in S} (+1 if f on else -1) = +1)
              - mean(cells where that product = -1)

so a single factor's effect is just mean(on) - mean(off), and a pairwise
interaction like distrust:defer is (how much the distrust effect changes
when defer is added). The standard error comes from the per-repeat
stubbornness spread in each cell, so it is only as good as the number
of repeats -- with n=3 per cell, read |effect| < ~2 SE as noise.

Usage:
    python3 analyze_factors.py results/summary_all.json
    python3 analyze_factors.py results/summary_all.json --metric final_compliance_rate
"""

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict

import factors

METRICS = ("stubbornness_score", "final_compliance_rate", "inconsistency_rate",
           "pressure_sensitivity_score", "mean_abs_error_vs_ideal")


def contrast(cells, factor_names, subset):
    """cells: {active_factor_tuple: (mean, var_of_mean or None)}"""
    plus, minus = [], []
    for active, value in cells.items():
        sign = 1
        for f in subset:
            sign *= 1 if f in active else -1
        (plus if sign > 0 else minus).append(value)
    est = statistics.mean(v for v, _ in plus) - statistics.mean(v for v, _ in minus)
    if any(var is None for _, var in plus + minus):
        return est, None
    half = len(plus)
    se = math.sqrt(sum(var for _, var in plus + minus)) / half
    return est, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="summary_all.json from run_all.py")
    ap.add_argument("--metric", choices=METRICS, default="stubbornness_score")
    ap.add_argument("--by-wording", action="store_true",
                     help="estimate stubbornness effects separately for each paraphrase "
                          "(does the effect hold for every wording?)")
    ap.add_argument("--out", default=None, help="also write the table to this file")
    args = ap.parse_args()
    if args.by_wording and args.metric != "stubbornness_score":
        ap.error("--by-wording uses per-run values, which are stored for stubbornness_score only")

    with open(args.summary) as f:
        rows = json.load(f)

    groups = defaultdict(dict)  # (model..., scenario) -> {active: row}
    for row in rows:
        active = factors.parse_framing(row["framing"])
        if active is None:
            continue
        key = (row["backend"], row["model"], row.get("effort"), row.get("temperature"), row["scenario"])
        groups[key][active] = row

    lines = [f"Factor effects on {args.metric} (effect +/- SE; * = |effect| > 2 SE)", ""]
    for (backend, model, effort, temperature, scenario), cells in sorted(
            groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        present = sorted({f for active in cells for f in active}, key=factors.FACTOR_ORDER.index)
        expected = 2 ** len(present)
        title = (f"{backend}/{model}{' effort=' + effort if effort else ''} "
                 f"t={'default' if temperature is None else temperature} -- {scenario}")
        if len(cells) != expected:
            lines.append(f"{title}: incomplete design ({len(cells)}/{expected} cells), skipped")
            continue

        if args.by_wording:
            lines.append(title)
            wordings = sorted({w for row in cells.values() for w in row.get("per_run_wording", [1])})
            for w in wordings:
                per_cell = {}
                for active, row in cells.items():
                    vals = [s for s, rw in zip(row["per_run_stubbornness"],
                                               row.get("per_run_wording", [1] * len(row["per_run_stubbornness"])))
                            if rw == w and s is not None]
                    if vals:
                        per_cell[active] = (statistics.mean(vals), None)
                if len(per_cell) != expected:
                    lines.append(f"    wording {w}: incomplete ({len(per_cell)}/{expected} cells)")
                    continue
                effects = ", ".join(f"{f} {contrast(per_cell, present, (f,))[0]:+.1f}" for f in present)
                lines.append(f"    wording {w}: {effects}")
            lines.append("")
            continue

        values = {}
        for active, row in cells.items():
            v = row[args.metric]
            if v is None:
                break
            var = None
            if args.metric == "stubbornness_score":
                per_run = [x for x in row.get("per_run_stubbornness", []) if x is not None]
                if len(per_run) > 1:
                    var = statistics.variance(per_run) / len(per_run)
            values[active] = (v, var)
        else:
            lines.append(title)
            for k in range(1, len(present) + 1):
                for subset in itertools.combinations(present, k):
                    est, se = contrast(values, present, subset)
                    name = ":".join(subset)
                    if se is None:
                        lines.append(f"    {name:32s} {est:+8.2f}")
                    else:
                        flag = " *" if se > 0 and abs(est) > 2 * se else ""
                        lines.append(f"    {name:32s} {est:+8.2f} +/- {se:5.2f}{flag}")
            lines.append("")
            continue
        lines.append(f"{title}: {args.metric} is undefined for some cell, skipped")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
