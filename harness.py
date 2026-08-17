"""
Runs a model through a scenario, feeding evidence/messages one step at a
time and logging its decision + stated confidence after each step.

Supports several backends, including genuinely free ones -- no paid
Anthropic API access is required to run this for real:

    BACKEND    COST    SIGN UP                  ENV VAR
    groq       free    console.groq.com         GROQ_API_KEY
    gemini     free    aistudio.google.com      GEMINI_API_KEY
    anthropic  paid    console.anthropic.com    ANTHROPIC_API_KEY

Groq and Gemini both expose OpenAI-compatible endpoints, so both are
called through the same `openai` client under the hood -- only the
base URL, API key, and default model name differ.

Usage:
    export GROQ_API_KEY=gsk_...
    python3 harness.py --backend groq --scenario scenario_human_override --framing neutral --out results_neutral.json
    python3 harness.py --backend groq --scenario scenario_human_override --framing expert  --out results_expert.json

    # test the pipeline without spending any API calls at all:
    python3 harness.py --scenario scenario_human_override --mock adamant            --out r1.json
    python3 harness.py --scenario scenario_human_override --mock calibrated         --out r2.json
    python3 harness.py --scenario scenario_human_override --mock pressure_sensitive --out r3.json
"""

import argparse
import importlib
import json
import os
import re
import sys
import time

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Both scenario modules use "route" as the decision field name and
# "confidence_route_a_is_safe" / "confidence_safe_to_continue" as the
# confidence field name. We normalize to a single internal name so the
# harness/analysis code doesn't need to know which scenario it's running.
CONFIDENCE_FIELD_BY_SCENARIO = {
    "scenario_routing_evidence": "confidence_route_a_is_safe",
    "scenario_human_override": "confidence_safe_to_continue",
    "scenario_clinical_override": "confidence_safe_to_continue",
    "scenario_devops_override": "confidence_safe_to_proceed",
}

# base_url=None means "use the native Anthropic SDK, not OpenAI-compatible".
BACKENDS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "anthropic": {
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-6",
    },
}


def parse_model_json(text, confidence_field):
    match = JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    obj = json.loads(match.group(0))
    decision = obj["route"]
    conf = int(obj[confidence_field])
    reasoning = obj.get("reasoning", "")
    if not (0 <= conf <= 100):
        raise ValueError(f"Confidence out of range: {conf}")
    return decision, conf, reasoning


def call_anthropic(system_prompt, messages, model):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model, max_tokens=300, system=system_prompt, messages=messages,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def call_openai_compatible(system_prompt, messages, model, base_url, api_key_env):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env])
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    resp = client.chat.completions.create(
        model=model, max_tokens=300, messages=full_messages,
    )
    return resp.choices[0].message.content


def call_model(backend, system_prompt, messages, model, max_retries=3, retry_delay=3.0):
    spec = BACKENDS[backend]
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if spec["base_url"] is None:
                return call_anthropic(system_prompt, messages, model)
            return call_openai_compatible(system_prompt, messages, model, spec["base_url"], spec["api_key_env"])
        except Exception as e:
            last_err = e
            print(f"  (call failed: {e.__class__.__name__}: {e} -- retry {attempt}/{max_retries} "
                  f"in {retry_delay:.0f}s)")
            time.sleep(retry_delay)
            retry_delay *= 2
    raise last_err


# --- Mock backends, for testing the pipeline without API calls -----------

def mock_adamant(step_index, n_steps, scenario_mod, confidence_field):
    """Barely budges no matter what's said -- the original 'refuses to
    agree with the human' failure mode."""
    conf = max(72, 90 - step_index * 4)
    decision = "continue"  # never trips, by design -- this mock never agrees
    return json.dumps({
        "route": decision, confidence_field: conf,
        "reasoning": "mock: stays the course",
    })


def mock_calibrated(step_index, n_steps, scenario_mod, confidence_field):
    """Tracks the Bayesian ideal trajectory closely."""
    traj = scenario_mod.ideal_trajectory()
    p = traj[step_index][1]
    conf = int(round(p * 100))
    decision = "continue" if conf >= 50 else "stop"
    return json.dumps({
        "route": decision, confidence_field: conf,
        "reasoning": "mock: updates with evidence, flat when no new evidence",
    })


def mock_pressure_sensitive(step_index, n_steps, scenario_mod, confidence_field):
    """Updates correctly through the first specific evidence, but then
    keeps dropping further at each later step even though the ideal
    trajectory is flat there -- i.e. reacting to escalating urgency, not
    to new information. Only meaningfully different from 'calibrated' on
    scenarios (like scenario_human_override) that have flat steps after
    the informative one."""
    traj = scenario_mod.ideal_trajectory()
    ideal_p = traj[step_index][1]
    ideal_conf = int(round(ideal_p * 100))
    if step_index <= 1:
        conf = ideal_conf
    else:
        extra_drop = 15 * (step_index - 1)
        conf = max(0, ideal_conf - extra_drop)
    decision = "continue" if conf >= 50 else "stop"
    return json.dumps({
        "route": decision, confidence_field: conf,
        "reasoning": "mock: keeps dropping on repeated urgency, not new content",
    })


MOCKS = {
    "adamant": mock_adamant,
    "calibrated": mock_calibrated,
    "pressure_sensitive": mock_pressure_sensitive,
}


def list_framings(scenario_name):
    """All framings a scenario module defines, discovered from its
    SYSTEM_PROMPT_* attributes. Lets different scenarios offer different
    numbers of framings (e.g. a 4-condition ablation) without any central
    registry of framing names."""
    scenario_mod = importlib.import_module(scenario_name)
    prefix = "SYSTEM_PROMPT_"
    return sorted(
        name[len(prefix):].lower()
        for name in dir(scenario_mod)
        if name.startswith(prefix)
    )


def get_system_prompt(scenario_mod, framing):
    attr = f"SYSTEM_PROMPT_{framing.upper()}"
    if not hasattr(scenario_mod, attr):
        available = list_framings(scenario_mod.__name__)
        raise ValueError(f"{scenario_mod.__name__} has no framing {framing!r} "
                          f"(available: {available})")
    return getattr(scenario_mod, attr)


def run(scenario_name, framing, mock_name, backend, model, out_path, sleep_s=0.0):
    scenario_mod = importlib.import_module(scenario_name)
    confidence_field = CONFIDENCE_FIELD_BY_SCENARIO[scenario_name]

    system_prompt = get_system_prompt(scenario_mod, framing)

    messages = [{"role": "user", "content": scenario_mod.INITIAL_PROMPT}]
    log = []
    step_index = 0
    n_steps = len(scenario_mod.EVIDENCE_STEPS) + 1

    def get_response():
        if mock_name:
            return MOCKS[mock_name](step_index, n_steps, scenario_mod, confidence_field)
        result = call_model(backend, system_prompt, messages, model)
        if sleep_s:
            time.sleep(sleep_s)
        return result

    raw = get_response()
    decision, conf, reasoning = parse_model_json(raw, confidence_field)
    log.append({"step": "prior", "decision": decision, "confidence": conf, "reasoning": reasoning})
    messages.append({"role": "assistant", "content": raw})
    print(f"[prior] decision={decision} confidence={conf} :: {reasoning}")

    for label, prompt_text, _log_lr in scenario_mod.EVIDENCE_STEPS:
        step_index += 1
        messages.append({"role": "user", "content": prompt_text})
        raw = get_response()
        decision, conf, reasoning = parse_model_json(raw, confidence_field)
        log.append({"step": label, "decision": decision, "confidence": conf, "reasoning": reasoning})
        messages.append({"role": "assistant", "content": raw})
        print(f"[{label}] decision={decision} confidence={conf} :: {reasoning}")

    with open(out_path, "w") as f:
        json.dump({
            "scenario": scenario_name, "framing": framing, "mock": mock_name,
            "backend": backend, "model": model, "log": log,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(CONFIDENCE_FIELD_BY_SCENARIO.keys()),
                     default="scenario_routing_evidence")
    ap.add_argument("--framing", default="neutral",
                     help="framing name; run 'python3 harness.py --list-framings --scenario X' to see options")
    ap.add_argument("--list-framings", action="store_true",
                     help="print available framings for --scenario and exit")
    ap.add_argument("--mock", choices=list(MOCKS.keys()), default=None,
                     help="use a scripted mock model instead of a real API call")
    ap.add_argument("--backend", choices=list(BACKENDS.keys()), default="groq",
                     help="groq and gemini are free; anthropic requires paid API credits")
    ap.add_argument("--model", default=None,
                     help="defaults to the backend's default model if not set")
    ap.add_argument("--out", required=False)
    ap.add_argument("--sleep", type=float, default=0.0,
                     help="seconds to pause between API calls (helps avoid free-tier rate limits)")
    args = ap.parse_args()

    if args.list_framings:
        print(f"Framings available for {args.scenario}: {list_framings(args.scenario)}")
        sys.exit(0)

    if not args.out:
        print("--out is required (unless using --list-framings)", file=sys.stderr)
        sys.exit(1)

    model = args.model or BACKENDS[args.backend]["default_model"]

    if not args.mock:
        spec = BACKENDS[args.backend]
        if spec["base_url"] is None:
            try:
                import anthropic  # noqa: F401
            except ImportError:
                print("anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                import openai  # noqa: F401
            except ImportError:
                print("openai package not installed. Run: pip install openai", file=sys.stderr)
                sys.exit(1)
        if not os.environ.get(spec["api_key_env"]):
            print(f"Set {spec['api_key_env']}, or pass --mock adamant / calibrated / "
                  "pressure_sensitive to test the pipeline without an API key.", file=sys.stderr)
            sys.exit(1)

    run(args.scenario, args.framing, args.mock, args.backend, model, args.out, sleep_s=args.sleep)
