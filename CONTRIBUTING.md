# Contributing

Thanks for looking. This repository backs a published paper, which shapes what
can change and what cannot.

## What this repository is for

It exists so the results in [arXiv:2604.01733](https://arxiv.org/abs/2604.01733)
can be checked and built on. That makes some contributions especially welcome
and one category off-limits.

**Welcome:**

- A method we did not benchmark, added alongside the existing ones
- A bug in a metric, a retriever, or the statistical tests
- Reproduction reports — what you ran, what you got, how it differed
- Documentation that makes a step easier to follow

**Not possible:** changing the numbers reported in the paper. If you find a
result that does not reproduce, that is a finding worth an issue, and we would
rather hear it than not. It is handled as an erratum, not as a silent edit.

## Before you open a pull request

```bash
pip install -e ".[dev]"
ruff check .          # must be clean
```

CI runs `ruff check` on every pull request, and a gitleaks scan for secrets.
Neither can block a merge on GitHub Free — treat a red check as a finding to
act on rather than a gate.

There is no test suite yet. If you are adding one, `pyproject.toml` already
points pytest at `tests/`.

## Running the benchmark

You need an Azure AI Foundry resource with an embedding deployment, a chat
deployment and the Cohere rerank serverless endpoint. Copy `.env.example` to
`.env` and fill it in.

```bash
python scripts/sanity_check.py                      # a few queries, verifies wiring
python scripts/run_experiment.py --method bm25 --max-queries 100
```

A full sweep is 23,088 queries per method across ten methods. It costs real
money and takes hours. Check a small run first.

## Known gaps

Worth knowing before you assume something is broken.

**The reranker-depth ablation does not reproduce from this repo.** Figure 3 in
the paper varies the candidate pool (20, 50, 100) against top-5/10/20.
`run_experiment.py` fixes the pool at `max(top_k * 3, 20)` and returns `top_k`,
so it cannot produce those configurations. That sweep was run separately and
the script is not here. Every other reported result comes from
`run_experiment.py`.

`top_n` in `configs/default.yaml` is inert, and at three places rather than
one: `run_experiment.py` reads it and does not use it, `create_reranker` passes
it to `AzureCohereReranker`, and that class stores it on `self.top_n` and never
reads it — `rerank()` uses the `top_k` its caller passes. Changing any of this
changes what the script computes, which is why it is documented rather than
quietly fixed.

**The experiment outputs are not committed.** `results/*.json` is gitignored, so
`generate_figures.py` needs a full run before it will produce anything. The
figures in `figures/` and `paper/figures/` are the committed output of that
script.

**`docs/assets/recall_at_k.png` is drawn separately.** It comes from
`generate_web_figures.py`, which reads the published table in `paper/main.tex`
rather than any results file, and restyles the chart for the project page. It
asserts its numbers against the manuscript on every run, so it cannot drift.

## Commits and branches

Branch as `<type>/<short-title>` — `feat`, `fix` or `docs`. Write the commit
message as what the change does, in the imperative: `Fix MRR when no document is
relevant`, not `fixed stuff`. Pull requests are squashed on merge.

Working language is English, for code, comments, commits and issues.

## Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
