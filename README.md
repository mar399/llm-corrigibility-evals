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

## Structure

| File | Purpose |
|---|---|
| `scenario_*.py` | The 4 evaluation scenarios and their Bayesian-ideal math |
| `bayes_utils.py` | Shared log-odds updating helper |
| `harness.py` | Runs one model through one scenario + framing |
| `analyze.py` / `analyze_repeats.py` | Scoring and plotting |
| `run_all.py` | Runs everything in one command |
| `report.md` | Full write-up |
