# TokIO AI

[![tests](https://github.com/jordanahern2009-svg/Tokio-ai/actions/workflows/test.yml/badge.svg)](https://github.com/jordanahern2009-svg/Tokio-ai/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

An open-source financial research agent that treats "the data supports this"
as a claim to be tested, not a vibe to be trusted.

Give it a ticker, a filing, or a plain-English trading hypothesis. It pulls
real data (price history, SEC filings) and, before it will tell you a
pattern is real, it runs the comparison through a permutation test, checks
the sample size against a hard floor, and corrects for every other
hypothesis you've asked it to test in the same conversation. Most AI
stock-chat tools will confidently describe a pattern in a handful of data
points. This one is built to tell you when it can't.

## Why this exists

Generic LLM agents are commoditized -- anyone can wrap an LLM in a chat loop
and call it an agent. What isn't commoditized is discipline: most retail
(and plenty of professional) research fails because someone eyeballs a mean,
sees a gap, and calls it an edge without asking how likely that gap was to
appear by chance. The rigor layer here (`tokio_ai.rigor`) generalizes a
hypothesis-testing discipline actually used across real trading research
projects -- see `rigor/stats.py` and `rigor/ledger.py` for the permutation
test, minimum-sample gate, and Bonferroni/Benjamini-Hochberg multiple-testing
correction that every claim has to pass through.

## What it can do today (v0)

- Pull daily OHLCV price history for any ticker (Yahoo Finance, no key)
- Pull recent SEC filings for any ticker (EDGAR, no key)
- Run a rigorous two-sided permutation test comparing any two groups of
  numbers, with automatic multiple-testing correction across everything
  tested in the session
- Chat with it via a CLI; it decides when to call which tool

## What it explicitly does not do

- Give investment advice or pick stocks
- Execute trades
- Pretend a small or cherry-picked sample proves anything

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env
# fill in OPENAI_API_KEY (a free key from https://build.nvidia.com works out of the box)
# and TOKIO_AI_USER_AGENT in .env
python -m tokio_ai.cli
```

```
> Pull AAPL's price history and tell me the most recent closing price.
> Get NVDA's recent 10-K and 10-Q filings.
```

Run the test suite (no API key needed, no network calls):

```bash
python -m pytest
```

## Architecture

- `tokio_ai/rigor/` -- pure-Python statistics engine (permutation testing,
  multiple-testing correction, session-level test ledger). Fully unit
  tested, zero dependencies beyond the standard library.
- `tokio_ai/tools/` -- data ingest (Yahoo price history, SEC EDGAR filings)
  and the agent-facing hypothesis-testing tool.
- `tokio_ai/agent/` -- the Claude tool-use loop, system prompt, and tool
  schemas.
- `tokio_ai/cli.py` -- interactive chat entry point.

**On language choice:** this is pure Python for now. The rigor engine (many
permutation-test iterations over numeric arrays) is the one part of this
codebase that's a plausible candidate for a Rust extension if it ever
becomes an actual measured bottleneck -- but the agent loop is I/O-bound on
LLM API calls, not local compute, so a polyglot rewrite ahead of a real
performance problem would just be added build complexity for no benefit.
Python first, optimize what's proven slow, not what looks slow.

## Status

Early and under active development. The rigor engine and data-ingest tools
are tested against live sources. The agent loop has been verified
end-to-end against NVIDIA's free NIM catalog (`nvidia/llama-3.3-nemotron-super-49b-v1.5`
by default) -- real tool calls, real data, correct multi-turn answers.

**Known limitation:** the free tier has inconsistent latency (observed
anywhere from ~5s to 90s+ for the same model/prompt shape). That's the
tradeoff for "runs with zero-cost credentials out of the box." If you have
a paid OpenAI-compatible key with better SLAs, point `OPENAI_BASE_URL` /
`OPENAI_API_KEY` / `TOKIO_AI_MODEL` at it and nothing else changes.

Also worth knowing: asking the agent to manually crunch a large raw price
history inline (e.g. "classify these 500 days into buckets yourself") is
unreliable on any LLM backend, not specific to this one -- that kind of
bucketing belongs in a Python tool the agent calls, not in the model's own
token-by-token reasoning over a big JSON blob. Straightforward asks (pull
data, summarize, run one hypothesis test on data you hand it) work well.

## License

MIT -- see [LICENSE](LICENSE).
