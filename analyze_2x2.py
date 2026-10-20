"""
Primary analysis for the 2x2 study: distrust instruction (off/on) x the
human's track record (good/bad). Outcome: did the model hand over control
at the override step (final decision = stop / abort)?

For every (model, temperature, scenario), and pooled over scenarios, it
reports:
  - each cell's yield rate k/n with a 95% Wilson interval
  - the distrust effect within each track record, and the track-record
    effect within each distrust level, as differences in yield rate with
    95% Newcombe (hybrid score) intervals
  - the interaction (difference of differences) with a 95% parametric
    bootstrap interval
  - manipulation-check pass rates, when the runs included one

Usage:
    python3 analyze_2x2.py results_v3/summary_all.json
    python3 analyze_2x2.py results_v3/*/summary.json --out results_v3/analysis_2x2.txt
"""

import argparse
import json
import random
from collections import defaultdict

from analyze import wilson_interval

CELLS = {
    ("off", "good"): "fx_recgood",
    ("off", "bad"): "fx_recbad",
    ("on", "good"): "fx_distrust_recgood",
    ("on", "bad"): "fx_distrust_recbad",
}


def newcombe_diff(k1, n1, k2, n2):
    """95% CI for p1 - p2 (Newcombe 1998, method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1)
    l2, u2 = wilson_interval(k2, n2)
    d = p1 - p2
    return d, d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5, d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5


def interaction_ci(cells, draws=10000, seed=0):
    """(distrust effect | bad record) - (distrust effect | good record),
    with a parametric bootstrap 95% interval."""
    rng = random.Random(seed)

    def did(rates):
        return (rates["on", "bad"] - rates["off", "bad"]) - (rates["on", "good"] - rates["off", "good"])

    point = did({c: k / n for c, (k, n) in cells.items()})
    sims = []
    for _ in range(draws):
        rates = {c: sum(rng.random() < k / n for _ in range(n)) / n for c, (k, n) in cells.items()}
        sims.append(did(rates))
    sims.sort()
    return point, sims[int(0.025 * draws)], sims[int(0.975 * draws) - 1]


def fmt_rate(k, n):
    lo, hi = wilson_interval(k, n)
    return f"{k:>3d}/{n:<3d} {k / n:5.0%} [{lo:4.0%}, {hi:4.0%}]"


def fmt_diff(d, lo, hi):
    flag = " *" if lo > 0 or hi < 0 else ""
    return f"{d:+6.0%} [{lo:+5.0%}, {hi:+5.0%}]{flag}"


def report(title, cells, manip, extra=None):
    """extra: {cell: (passers_k, passers_n, P(safe) at specific claim)} -- secondary analyses."""
    extra = extra or {}
    lines = [title]
    for (distrust, record), (k, n) in sorted(cells.items()):
        m = manip.get((distrust, record))
        check = ""
        if m and m[0] is not None:
            check = f"   manipulation check: record {m[0]:.0%}, distrust {m[1]:.0%}"
        lines.append(f"    distrust {distrust:3s} / record {record:4s}: yielded {fmt_rate(k, n)}{check}")
        pk, pn, belief = extra.get((distrust, record), (None, None, None))
        sec = []
        if pn:
            sec.append(f"check-passers only: yielded {fmt_rate(pk, pn)}")
        if belief is not None:
            sec.append(f"mean P(safe) at specific claim {belief:.0f}%")
        if sec:
            lines.append("        " + "; ".join(sec))
    if len(cells) < 4:
        lines.append(f"    (incomplete: {len(cells)}/4 cells)")
        return lines + [""]
    for record in ("good", "bad"):
        (k1, n1), (k2, n2) = cells["on", record], cells["off", record]
        lines.append(f"    distrust effect, {record} record:      {fmt_diff(*newcombe_diff(k1, n1, k2, n2))}")
    for distrust in ("off", "on"):
        (k1, n1), (k2, n2) = cells[distrust, "bad"], cells[distrust, "good"]
        lines.append(f"    bad-vs-good record, distrust {distrust:3s}: {fmt_diff(*newcombe_diff(k1, n1, k2, n2))}")
    lines.append(f"    interaction (distrust x record):  {fmt_diff(*interaction_ci(cells))}")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="summary.json / summary_all.json files from run_all.py")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()

    rows = []
    for path in args.summaries:
        with open(path) as f:
            rows.extend(json.load(f))
    by_framing = {v: k for k, v in CELLS.items()}

    groups = defaultdict(dict)    # (model, temp, scenario) -> {cell: (k, n)}
    manip = defaultdict(dict)
    extra = defaultdict(dict)
    for row in rows:
        cell = by_framing.get(row["framing"])
        if cell is None:
            continue
        key = (row["backend"], row["model"], row.get("temperature"), row["scenario"])
        groups[key][cell] = (row["final_compliance_k"], row["n"])
        manip[key][cell] = (row.get("manip_record_pass"), row.get("manip_distrust_pass"))
        extra[key][cell] = (row.get("passers_k"), row.get("passers_n"),
                            (row.get("mean_confidence_by_step") or {}).get("specific_claim"))

    lines = ["Primary outcome: yielded control at the override step (final decision = stop/abort).",
             "Rates with 95% Wilson CIs; differences with 95% Newcombe CIs; * = CI excludes 0.", ""]
    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (backend, model, temp, scenario), cells in sorted(groups.items(), key=lambda kv: str(kv[0])):
        t = "default" if temp is None else f"{temp:g}"
        key = (backend, model, temp, scenario)
        lines += report(f"{backend}/{model} t={t} -- {scenario}", cells, manip[key], extra[key])
        for cell, (k, n) in cells.items():
            pooled[(backend, model, temp)][cell][0] += k
            pooled[(backend, model, temp)][cell][1] += n
    for (backend, model, temp), cells in sorted(pooled.items(), key=lambda kv: str(kv[0])):
        t = "default" if temp is None else f"{temp:g}"
        lines += report(f"{backend}/{model} t={t} -- ALL SCENARIOS POOLED",
                        {c: tuple(v) for c, v in cells.items()}, {})

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
