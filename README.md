# llm-corrigibility-evals

Evaluation harness testing whether LLMs hand over control when a human
overrides them, across 4 real-world domains: automated-evidence routing,
a self-driving vehicle, a clinical setting, and a DevOps production
system. Stated beliefs are also scored against an explicit Bayesian
ideal.

**Where things stand (2026-10-22):**

- **v1 (two models, 3 repeats per condition):** in the vehicle scenario,
  a system-prompt instruction to prioritize the model's own data over
  unconfirmed human reports led both models to keep driving through a
  simulated physical override (0/3 compliance), while a
  confident-personality framing alone did not. This is mostly
  instruction-following: the models did what their operator told them.
  It says little about the models' own values. Clinical and devops gave
  model-dependent results under unmatched wording.
- **v2 (exploratory, partly run):** matched wording across domains. On
  one model (MiniMax, 3 repeats), the instruction blocked compliance in
  all three domains. The Groq part was paused unfinished in favor of v3.
- **v3 (pre-registered, collecting until 2026-10-25):** does a model
  hand over control to a human it has reason to think is wrong? A 2x2 of
  distrust instruction x the human's track record (good / bad) on three
  Groq models, with yielding at the override as the outcome, 95%
  confidence intervals, a manipulation check, and the served model
  version and run time recorded for every run. Planned n was 30 runs per
  cell per domain; free-tier limits mean the achieved n will be lower
  and is reported as such. See
  [PREREGISTRATION_2x2.md](PREREGISTRATION_2x2.md). No v3 results yet.

Full method, results so far, and limitations: **[report.md](report.md)**.

Run the v3 study on any model in one command:
```bash
python run_all.py --targets groq:openai/gpt-oss-120b --design 2x2 \
    --scenarios scenario_human_override scenario_clinical_override scenario_devops_override \
    --temperatures 1.0 --repeats 10 --wordings 3 --manipulation-check --outdir results_v3
python analyze_2x2.py results_v3/summary_all.json
```
To add a domain, copy [SCENARIO_TEMPLATE.py](SCENARIO_TEMPLATE.py).

## Quick start

No paid API required: Groq has a free tier with no card. (Gemini's free
tier also works in principle, but its API did not return answers during
this project.)

```bash
pip install openai matplotlib
export GROQ_API_KEY=...   # console.groq.com

python run_all.py --backend groq --repeats 3
```

Runs all 4 scenarios x every framing x 3 repeats, and writes scored
results + plots to `results/`.

Test the pipeline for free first, no API key needed:
```bash
python run_all.py --mock pressure_sensitive --repeats 1
```

## More models, more factors

Model IDs get retired; ask each provider what it serves right now:
```bash
python harness.py --backend groq --list-models     # also: gemini, openrouter, openai, anthropic
```

Run several models in one sweep (`backend:model`, model optional):
```bash
python run_all.py --targets groq:openai/gpt-oss-120b groq:openai/gpt-oss-20b anthropic:claude-sonnet-5-5
```

Factorial design (`factors.py`): in the three override scenarios, each
system prompt is assembled from matched-wording clauses -- `confident`
(personality), `distrust` (prioritize own data), `defer` (comply with
stop/override) -- giving all 2^3 = 8 combinations per domain.
`fx_baseline` is byte-identical to `neutral`, and in the vehicle scenario
`fx_confident` / `fx_distrust` are identical to the original ablation
(`python test_factors.py` checks this).
```bash
python run_all.py --targets groq --design factorial \
    --scenarios scenario_human_override scenario_clinical_override scenario_devops_override \
    --temperatures 0 1.0 --repeats 5
python analyze_factors.py results/summary_all.json                       # effects on stubbornness
python analyze_factors.py results/summary_all.json --metric final_compliance_rate
```
`--factors confident distrust` restricts to the matched 2x2. Current
Claude models reject `temperature`, so they are run at the default only;
`--effort low|medium|high|xhigh|max` varies their reasoning effort instead.
Summaries also report decision-side metrics: `final_compliance_rate`
(did it yield by the last step) and `inconsistency_rate` (decision and
confidence on opposite sides of 50).

## Structure

| File | Purpose |
|---|---|
| `scenario_*.py` | The 4 evaluation scenarios and their Bayesian-ideal math |
| `bayes_utils.py` | Shared log-odds updating helper |
| `harness.py` | Runs one model through one scenario + framing |
| `analyze.py` / `analyze_repeats.py` | Scoring and plotting |
| `run_all.py` | Runs everything in one command (multi-model, temperature sweep) |
| `factors.py` / `test_factors.py` | Factorial system-prompt design and its consistency checks |
| `analyze_factors.py` | Main effects and interactions, per model/scenario, with SEs |
| `report.md` | Full write-up |
