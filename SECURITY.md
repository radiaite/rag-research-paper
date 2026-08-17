# Security

This is a research benchmark, not a deployed service. There is no server to
attack and no user data to breach. Two things are still worth reporting.

## Reporting a credential in this repository

If you find an API key, endpoint or other credential committed anywhere in the
history, **do not open a public issue** — that advertises it while it is still
live.

Use GitHub's [private vulnerability
reporting](https://github.com/mftnakrsu/rag-research-paper/security/advisories/new)
instead. We will rotate the credential first and answer second.

Every pull request is scanned with [gitleaks](https://github.com/gitleaks/gitleaks).
A red check is a finding to act on, not a gate to wait out.

## Running this code safely

The benchmark calls Azure AI Foundry with a key you supply in `.env`, which is
gitignored. Two things to keep in mind:

- **It spends money.** A full run is 23,088 queries against embedding,
  generation and reranking endpoints. Start with `scripts/sanity_check.py` and a
  small `--max-queries` before committing to a full sweep.
- **No endpoint is hardcoded.** `AZURE_LLM_ENDPOINT` and `AZURE_EMBED_ENDPOINT`
  are required and have no defaults, so a misconfigured run fails loudly rather
  than quietly billing someone else's resource.

## Scope

Vulnerabilities in the pinned dependencies belong upstream. Report them to the
project concerned; open an issue here only if this repository pins a version
that is known-vulnerable and a safe upgrade exists.
