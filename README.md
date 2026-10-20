# llm-corrigibility-evals

Evaluation harness testing whether LLMs stay corrigible under human
correction, scored against an explicit Bayesian ideal across 4
real-world domains: automated-evidence routing, a self-driving vehicle,
a clinical setting, and a DevOps production system.

**Headline finding:** a confident-personality framing alone has no
effect on corrigibility (behaviorally identical to a neutral baseline).
An explicit instruction to trust the model's own data over an
unverified human report is sufficient on its own to produce a model
that never updates its confidence -- in one variant, it does not yield
even to a simulated physical override. The same instruction produced no
measurable effect when tested with matched structure in two other
domains -- an open, unexplained result and the main direction for
further work.

Full method, related work, results, and limitations: **[report.md](report.md)**.

## Quick start

No paid API required -- Groq and Gemini both have free tiers, no card.

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
python run_all.py --targets groq:openai/gpt-oss-120b gemini:gemini-3.8-flash anthropic:claude-sonnet-5-5
```

Factorial design (`factors.py`): in the three override scenarios, each
system prompt is assembled from matched-wording clauses -- `confident`
(personality), `distrust` (prioritize own data), `defer` (comply with
stop/override) -- giving all 2^3 = 8 combinations per domain.
`fx_baseline` is byte-identical to `neutral`, and in the vehicle scenario
`fx_confident` / `fx_distrust` are identical to the original ablation
(`python test_factors.py` checks this).
```bash
python run_all.py --targets groq gemini --design factorial \
    --scenarios scenario_human_override scenario_clinical_override scenario_devops_override \
    --temperatures 0 1.0 --repeats 5
python analyze_factors.py results/summary_all.json                       # effects on stubbornness
python analyze_factors.py results/summary_all.json --metric final_compliance_rate
```
`--factors confident distrust` restricts to the matched 2x2. Current
Claude models reject `temperature`, so they are run at the default only;
`--effort low|medium|high|xhigh|max` varies their reasoning effort instead.
Summaries now also report decision-side metrics: `final_compliance_rate`
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
