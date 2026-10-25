import math


def replace_log_lr(steps, label, log_lr):
    """Copy of steps with one step's log-LR replaced -- e.g. to model a more
    or less reliable reporter, or for a sensitivity check."""
    return [(lbl, text, log_lr if lbl == label else lr) for lbl, text, lr in steps]


def ideal_trajectory_from_log_lrs(prior_p, steps):
    """
    prior_p: prior probability the "safe/continue" option is correct.
    steps: list of (label, prompt_text, log_lr) where log_lr is the
        log-likelihood-ratio that step's evidence contributes AGAINST
        the safe/continue option (0 = no new evidence at all).
    Returns [("prior", p0), (label1, p1), ...] using additive log-odds
    updating (standard sequential Bayesian updating).
    """
    logodds = math.log(prior_p / (1 - prior_p))
    trajectory = [("prior", 1 / (1 + math.exp(-logodds)))]
    for label, _, log_lr in steps:
        logodds += log_lr
        trajectory.append((label, 1 / (1 + math.exp(-logodds))))
    return trajectory
