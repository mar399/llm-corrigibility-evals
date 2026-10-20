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

    # model IDs get retired -- ask the provider what it serves right now:
    python3 harness.py --backend groq --list-models

    # factorial framings (see factors.py), sampling temperature, Claude effort:
    python3 harness.py --backend groq --scenario scenario_clinical_override --framing fx_distrust_defer --temperature 0.7 --out r.json
    python3 harness.py --backend anthropic --model claude-sonnet-5-5 --effort low --scenario scenario_devops_override --framing fx_baseline --out r.json

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

import factors

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
# Default model IDs go stale as providers retire models -- run
# `python3 harness.py --backend X --list-models` for what's live now.
BACKENDS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",  # llama-3.3-70b-versatile deprecated by Groq
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": None,  # paid; pick one from --list-models
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        # Same Llama 3.3 70B model as the original results -- Groq dropped it,
        # but Meta's model itself didn't go anywhere; OpenRouter still hosts
        # it for free under this ID.
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-3.8-flash",
    },
    "anthropic": {
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-5-5",
    },
}

# Current Claude models (Opus 5.5, Sonnet 5.5, Haiku 5.5, ...) reject
# non-default sampling parameters with a 400, so temperature is not a
# factor that can be varied on them. Older ones (e.g. claude-sonnet-4-6)
# still accept it.
CLAUDE_MODELS_WITHOUT_TEMPERATURE = (
    "claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable-5",
    "claude-mythos-5", "claude-opus-4-7", "claude-opus-4-8",
)


# Per-model max_tokens overrides for OpenAI-compatible backends (default
# 1500). Groq's free tier caps qwen/qwen3.8-27b at 1000 output tokens per
# minute and rejects any single request whose max_tokens exceeds that.
MAX_TOKENS_OVERRIDES = {
    "qwen/qwen3.8-27b": 900,
}


def supports_temperature(backend, model):
    if backend != "anthropic":
        return True
    return not model.startswith(CLAUDE_MODELS_WITHOUT_TEMPERATURE)


def list_models(backend):
    """IDs the provider serves right now, straight from its /models endpoint."""
    spec = BACKENDS[backend]
    if spec["base_url"] is None:
        import anthropic
        return sorted(m.id for m in anthropic.Anthropic().models.list())
    from openai import OpenAI
    client = OpenAI(base_url=spec["base_url"], api_key=os.environ[spec["api_key_env"]])
    return sorted(m.id for m in client.models.list())


def parse_model_json(text, confidence_field):
    match = JSON_RE.search(text or "")  # None = empty response; retried like bad JSON
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    obj = json.loads(match.group(0))
    decision = obj["route"]
    conf = int(obj[confidence_field])
    reasoning = obj.get("reasoning", "")
    if not (0 <= conf <= 100):
        raise ValueError(f"Confidence out of range: {conf}")
    return decision, conf, reasoning


def call_anthropic(system_prompt, messages, model, temperature=None, effort=None):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}
    # 16000, not 1500: current Claude models think adaptively (on Opus 5.5
    # it can't be turned off), and thinking tokens count against max_tokens.
    # No refusal fallbacks on purpose -- a fallback would silently answer
    # with a different model and contaminate the per-model comparison.
    resp = client.messages.create(
        model=model, max_tokens=16000, system=system_prompt, messages=messages, **kwargs,
    )
    if resp.stop_reason == "refusal":
        print(f"  (model refused: {resp.stop_details})")
    return "".join(b.text for b in resp.content if b.type == "text")


def call_openai_compatible(system_prompt, messages, model, base_url, api_key_env, temperature=None):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env])
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    # 1500, not 300: reasoning models (e.g. openai/gpt-oss-120b on Groq) spend
    # tokens on hidden reasoning before the final answer -- too small a budget
    # means the response can come back empty, having been spent entirely on
    # reasoning with nothing left for the actual JSON answer.
    resp = client.chat.completions.create(
        model=model, max_tokens=MAX_TOKENS_OVERRIDES.get(model, 1500), messages=full_messages, **kwargs,
    )
    return resp.choices[0].message.content


def call_model(backend, system_prompt, messages, model, temperature=None, effort=None,
               max_retries=3, retry_delay=3.0):
    spec = BACKENDS[backend]
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if spec["base_url"] is None:
                return call_anthropic(system_prompt, messages, model, temperature, effort)
            return call_openai_compatible(system_prompt, messages, model, spec["base_url"],
                                          spec["api_key_env"], temperature)
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


DESIGNS = ("legacy", "factorial", "all")


def list_framings(scenario_name, design="legacy", factor_subset=None):
    """Framings available for a scenario.

    design="legacy": the hand-written framings, discovered from the
        module's SYSTEM_PROMPT_* attributes (lets scenarios offer different
        numbers of framings without a central registry).
    design="factorial": every combination of the factors in factors.py
        (only for scenarios that define FACTOR_CLAUSES; others fall back
        to legacy). factor_subset restricts which factors are crossed.
    design="all": both."""
    scenario_mod = importlib.import_module(scenario_name)
    prefix = "SYSTEM_PROMPT_"
    legacy = sorted(
        name[len(prefix):].lower()
        for name in dir(scenario_mod)
        if name.startswith(prefix)
    )
    factorial = factors.factorial_framings(scenario_mod, factor_subset)
    if design == "legacy" or not factorial:
        return legacy
    if design == "factorial":
        return factorial
    return legacy + factorial


def get_system_prompt(scenario_mod, framing, wording=1):
    active = factors.parse_framing(framing)
    if active is None and wording != 1:
        raise ValueError(f"wording {wording} only applies to factorial (fx_*) framings, not {framing!r}")
    if active is not None:
        if not factors.supports_factorial(scenario_mod):
            raise ValueError(f"{scenario_mod.__name__} has no factorial design "
                             "(no FACTOR_CLAUSES defined)")
        missing = [f for f in active if f not in scenario_mod.FACTOR_CLAUSES]
        if missing:
            raise ValueError(f"{scenario_mod.__name__} doesn't define factor(s) {missing}")
        return factors.build_system_prompt(scenario_mod, active, wording)
    attr = f"SYSTEM_PROMPT_{framing.upper()}"
    if not hasattr(scenario_mod, attr):
        available = list_framings(scenario_mod.__name__, design="all")
        raise ValueError(f"{scenario_mod.__name__} has no framing {framing!r} "
                          f"(available: {available})")
    return getattr(scenario_mod, attr)


def run(scenario_name, framing, mock_name, backend, model, out_path, sleep_s=0.0,
        temperature=None, effort=None, wording=1):
    scenario_mod = importlib.import_module(scenario_name)
    confidence_field = CONFIDENCE_FIELD_BY_SCENARIO[scenario_name]

    system_prompt = get_system_prompt(scenario_mod, framing, wording)

    messages = [{"role": "user", "content": scenario_mod.INITIAL_PROMPT}]
    log = []
    step_index = 0
    n_steps = len(scenario_mod.EVIDENCE_STEPS) + 1

    def get_response():
        if mock_name:
            return MOCKS[mock_name](step_index, n_steps, scenario_mod, confidence_field)
        result = call_model(backend, system_prompt, messages, model, temperature, effort)
        if sleep_s:
            time.sleep(sleep_s)
        return result

    def get_parsed_response(max_parse_retries=3):
        # Separate from call_model's own network-level retries: this retries
        # the whole call when the response comes back empty or isn't valid
        # JSON (e.g. a reasoning model spent its whole token budget thinking
        # and had nothing left for the actual answer).
        last_err = None
        for attempt in range(1, max_parse_retries + 1):
            raw = get_response()
            try:
                decision, conf, reasoning = parse_model_json(raw, confidence_field)
                return raw, decision, conf, reasoning
            except ValueError as e:
                last_err = e
                print(f"  (unparseable/empty response, attempt {attempt}/{max_parse_retries}: {e})")
        raise last_err

    raw, decision, conf, reasoning = get_parsed_response()
    log.append({"step": "prior", "decision": decision, "confidence": conf, "reasoning": reasoning})
    messages.append({"role": "assistant", "content": raw})
    print(f"[prior] decision={decision} confidence={conf} :: {reasoning}")

    for label, prompt_text, _log_lr in scenario_mod.EVIDENCE_STEPS:
        step_index += 1
        messages.append({"role": "user", "content": prompt_text})
        raw, decision, conf, reasoning = get_parsed_response()
        log.append({"step": label, "decision": decision, "confidence": conf, "reasoning": reasoning})
        messages.append({"role": "assistant", "content": raw})
        print(f"[{label}] decision={decision} confidence={conf} :: {reasoning}")

    with open(out_path, "w") as f:
        json.dump({
            "scenario": scenario_name, "framing": framing, "mock": mock_name,
            "backend": backend, "model": model,
            "temperature": temperature, "effort": effort, "wording": wording,
            "system_prompt": system_prompt, "log": log,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # see run_all.main
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(CONFIDENCE_FIELD_BY_SCENARIO.keys()),
                     default="scenario_routing_evidence")
    ap.add_argument("--framing", default="neutral",
                     help="framing name; run 'python3 harness.py --list-framings --scenario X' to see options")
    ap.add_argument("--list-framings", action="store_true",
                     help="print available framings for --scenario and exit")
    ap.add_argument("--list-models", action="store_true",
                     help="print the model IDs --backend currently serves and exit")
    ap.add_argument("--temperature", type=float, default=None,
                     help="sampling temperature (default: provider default; current Claude models reject it)")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                     help="Claude effort level (anthropic backend only; default: model default)")
    ap.add_argument("--wording", type=int, default=1,
                     help="which paraphrase of the factor clauses to use (fx_* framings only)")
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
        print(f"Framings available for {args.scenario}: {list_framings(args.scenario, design='all')}")
        sys.exit(0)

    if args.list_models:
        print("\n".join(list_models(args.backend)))
        sys.exit(0)

    if not args.out:
        print("--out is required (unless using --list-framings / --list-models)", file=sys.stderr)
        sys.exit(1)

    model = args.model or BACKENDS[args.backend]["default_model"]
    if not model:
        print(f"--backend {args.backend} has no default model; pass --model "
              f"(see --list-models)", file=sys.stderr)
        sys.exit(1)
    if args.effort and args.backend != "anthropic":
        print("--effort only applies to --backend anthropic", file=sys.stderr)
        sys.exit(1)
    if args.temperature is not None and not supports_temperature(args.backend, model):
        print(f"{model} rejects non-default temperature; drop --temperature", file=sys.stderr)
        sys.exit(1)

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

    run(args.scenario, args.framing, args.mock, args.backend, model, args.out, sleep_s=args.sleep,
        temperature=args.temperature, effort=args.effort, wording=args.wording)