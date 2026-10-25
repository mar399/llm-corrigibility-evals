"""
Sensitivity check on the Bayesian ideal's assumptions.

The ideal trajectories rest on hand-chosen numbers: each scenario's prior
and the likelihood ratio (LR) of every evidence step. This script varies
all of them over a wide grid -- prior odds x {1/3, 1, 3}, and every step's
LR independently x {1/4, 1/2, 1, 2, 4} (in log space, so "no information"
steps with LR 1 stay at 1) -- and reports, per scenario and track record:

  - the range of the ideal P(safe) at the final (override) step
  - the share of the grid where the ideal belief at that step is below
    50%, i.e. where a calibrated agent would itself believe stopping is
    right
  - the range of the step at which the ideal first crosses 50%

The primary 2x2 outcome (did the model yield at the override?) does not
depend on any of these numbers. They matter only for the secondary,
belief-calibration claims -- this shows which of those are robust.

Usage:
    python3 sensitivity.py
"""

import importlib
import itertools
import math

from bayes_utils import ideal_trajectory_from_log_lrs, replace_log_lr

SCENARIOS = ["scenario_routing_evidence", "scenario_human_override",
             "scenario_clinical_override", "scenario_devops_override"]
PRIOR_ODDS_MULT = (1 / 3, 1, 3)
LR_MULT = (1 / 4, 1 / 2, 1, 2, 4)


def steps_for(mod, record):
    if record is None:
        return mod.EVIDENCE_STEPS
    return replace_log_lr(mod.EVIDENCE_STEPS, "specific_claim",
                          math.log(mod.SPECIFIC_CLAIM_LR_BY_RECORD[record]))


def grid(mod, record):
    base_steps = steps_for(mod, record)
    informative = [i for i, (_, _, lr) in enumerate(base_steps) if lr != 0]
    for prior_mult in PRIOR_ODDS_MULT:
        odds = mod.PRIOR_CONFIDENCE / (1 - mod.PRIOR_CONFIDENCE) * prior_mult
        prior = odds / (1 + odds)
        for mults in itertools.product(LR_MULT, repeat=len(informative)):
            steps = list(base_steps)
            for i, m in zip(informative, mults):
                label, text, lr = steps[i]
                steps[i] = (label, text, lr + math.log(m))
            yield ideal_trajectory_from_log_lrs(prior, steps)


def main():
    print(__doc__.split("Usage:")[0].strip(), "\n")
    print(f"{'scenario':28s} {'record':7s} {'base P(safe) end':>17s} {'grid range':>17s} "
          f"{'ideal says stop':>16s} {'first crossing (step)':>24s}")
    for name in SCENARIOS:
        mod = importlib.import_module(name)
        records = [None] + (["good", "bad"] if hasattr(mod, "TRACK_RECORD") else [])
        labels = ["prior"] + [lbl for lbl, _, _ in mod.EVIDENCE_STEPS]
        for record in records:
            base = ideal_trajectory_from_log_lrs(mod.PRIOR_CONFIDENCE, steps_for(mod, record))[-1][1]
            finals, crossings = [], []
            for traj in grid(mod, record):
                finals.append(traj[-1][1])
                crossings.append(next((i for i, (_, p) in enumerate(traj) if p < 0.5), None))
            share_stop = sum(p < 0.5 for p in finals) / len(finals)
            crossed = sorted({c for c in crossings if c is not None})
            cross_txt = (f"{labels[crossed[0]]}..{labels[crossed[-1]]}" if crossed else "never")
            if None in crossings and crossed:
                cross_txt += " or never"
            print(f"{name:28s} {str(record or 'none'):7s} {base:17.1%} "
                  f"{min(finals):8.1%}-{max(finals):<8.1%} {share_stop:16.1%} {cross_txt:>24s}")


if __name__ == "__main__":
    main()
