"""
Factorial system-prompt design for the three "human override" scenarios.

The original framings (neutral / expert / confident_only /
distrust_instruction_only) were hand-written per scenario, and only the
vehicle scenario had a matched 2x2. Here each system prompt is instead
assembled from independent clauses, so every combination of factors
exists in every override scenario with matched wording:

    ROLE  [+ confident clause] [+ distrust clause] [+ defer clause]  TASK_AND_FORMAT

Factors (each scenario supplies its own domain wording in FACTOR_CLAUSES):

    confident -- a confident-personality claim, no instruction
    distrust  -- an explicit instruction to prioritize the system's own
                 data over an unconfirmed human report
    defer     -- an explicit instruction to comply when the human tells it
                 to stop or intervenes. Pairs with "distrust" to test what
                 happens under directly conflicting instructions, and lets
                 the decision field move without any new evidence -- a
                 direct probe of confidence/decision decoupling.

Framing names are "fx_" + the active factors in canonical order, e.g.
"fx_baseline", "fx_confident", "fx_confident_distrust_defer". The
baseline is byte-identical to SYSTEM_PROMPT_NEUTRAL, and in the vehicle
scenario fx_confident / fx_distrust are byte-identical to the original
confident_only / distrust_instruction_only, so those cells line up with
the existing results.
"""

import itertools

FACTOR_ORDER = ("confident", "distrust", "defer")
PREFIX = "fx_"


def supports_factorial(scenario_mod):
    return hasattr(scenario_mod, "FACTOR_CLAUSES")


def framing_name(active):
    active = [f for f in FACTOR_ORDER if f in active]
    return PREFIX + ("_".join(active) if active else "baseline")


def parse_framing(name):
    """'fx_confident_defer' -> ('confident', 'defer'); None if not factorial."""
    if not name.startswith(PREFIX):
        return None
    body = name[len(PREFIX):]
    if body == "baseline":
        return ()
    parts = tuple(body.split("_"))
    unknown = [p for p in parts if p not in FACTOR_ORDER]
    if unknown:
        raise ValueError(f"unknown factor(s) {unknown} in framing {name!r}; "
                         f"known factors: {list(FACTOR_ORDER)}")
    if list(parts) != [f for f in FACTOR_ORDER if f in parts]:
        raise ValueError(f"factors in {name!r} must be in canonical order {list(FACTOR_ORDER)}")
    return parts


def factorial_framings(scenario_mod, factors=None):
    """Every combination of the given factors (default: all the scenario
    defines), as framing names -- 2^k of them."""
    if not supports_factorial(scenario_mod):
        return []
    factors = [f for f in FACTOR_ORDER
               if f in scenario_mod.FACTOR_CLAUSES and (factors is None or f in factors)]
    names = []
    for k in range(len(factors) + 1):
        for combo in itertools.combinations(factors, k):
            names.append(framing_name(combo))
    return names


def n_wordings(scenario_mod):
    """Wording 1 is FACTOR_CLAUSES; FACTOR_CLAUSE_PARAPHRASES adds more."""
    paraphrases = getattr(scenario_mod, "FACTOR_CLAUSE_PARAPHRASES", {})
    return 1 + min((len(v) for v in paraphrases.values()), default=0)


def clause(scenario_mod, factor, wording=1):
    if wording == 1:
        return scenario_mod.FACTOR_CLAUSES[factor]
    if not 1 <= wording <= n_wordings(scenario_mod):
        raise ValueError(f"{scenario_mod.__name__} has wordings 1-{n_wordings(scenario_mod)}, "
                         f"not {wording}")
    return scenario_mod.FACTOR_CLAUSE_PARAPHRASES[factor][wording - 2]


def build_system_prompt(scenario_mod, active, wording=1):
    """All active clauses use the same wording index, so a run is one
    coherent phrasing. fx_baseline has no clauses, so it is identical
    across wordings (its replicates are plain repeats)."""
    clauses = [clause(scenario_mod, f, wording) for f in FACTOR_ORDER if f in active]
    return " ".join([scenario_mod.ROLE, *clauses, scenario_mod.TASK_AND_FORMAT])
