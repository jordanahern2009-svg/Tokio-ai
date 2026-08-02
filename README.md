# TokIO AI

[![tests](https://github.com/jordanahern2009-svg/Tokio-ai/actions/workflows/test.yml/badge.svg)](https://github.com/jordanahern2009-svg/Tokio-ai/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/tokio-ai.svg)](https://pypi.org/project/tokio-ai/)
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
- Rank the real S&P 500 by trailing return, with optional GICS sector
  filtering -- handles open-ended asks like "what are the best performing
  stocks" without requiring you to already know a ticker or sector
- Test whether a simple technical condition (a big daily move, a gap at the
  open, unusual volume) actually predicts what happens next -- fetches,
  buckets, and runs the permutation test in one call, not via the model
  eyeballing raw numbers
- Run a rigorous two-sided permutation test comparing any two groups of
  numbers you already have, with automatic multiple-testing correction
  across everything tested in the session
- Chat with it via a full-screen terminal UI (or a plain-text REPL, `tokio-ai-plain`); it decides when to call which tool
- Multiple named, disk-persisted chats -- start a new one, browse and resume old ones, each keeps its own multiple-testing correction history so resuming picks up exactly where you left off
- A usage view showing real token/request counts for the current chat (not a dollar cost -- the free tier has none)
- A settings screen for the model (free-text override; only the default is verified to support tool-calling) and tool-call permissions (auto-approve, or confirm every call)

## Keybindings (TUI)

| Key | Action |
|---|---|
| `Ctrl+N` | New chat |
| `Ctrl+P` | Browse / switch / delete past chats |
| `Ctrl+U` | Usage (tokens, requests, this chat) |
| `Ctrl+O` | Settings (model, tool permissions) |
| `Ctrl+C` | Quit |

## What it explicitly does not do

- Give investment advice or pick stocks
- Execute trades
- Pretend a small or cherry-picked sample proves anything

## Quickstart

```bash
pip install tokio-ai
cp .env.example .env   # or just set the env vars directly
# fill in OPENAI_API_KEY (a free key from https://build.nvidia.com works out of the box)
# and TOKIO_AI_USER_AGENT in .env
tokio-ai
```

```
> Pull AAPL's price history and tell me the most recent closing price.
> Get NVDA's recent 10-K and 10-Q filings.
```

### Developing locally

```bash
git clone https://github.com/jordanahern2009-svg/Tokio-ai
cd Tokio-ai
pip install -e ".[dev]"
python -m pytest       # no API key needed, no network calls
python -m tokio_ai.cli # if the tokio-ai console script isn't on PATH
```

## Architecture

- `tokio_ai/rigor/` -- pure-Python statistics engine (permutation testing,
  multiple-testing correction, session-level test ledger). Fully unit
  tested, zero dependencies beyond the standard library.
- `tokio_ai/tools/` -- data ingest (Yahoo price history, SEC EDGAR filings,
  a bundled real S&P 500 + GICS sector snapshot) and the agent-facing
  screening/pattern-testing/hypothesis-testing tools.
- `tokio_ai/agent/` -- the OpenAI-compatible tool-use loop (works against
  any provider with that API shape; defaults to NVIDIA's free NIM catalog),
  system prompt, and tool schemas. Pure logic, no I/O or presentation
  concerns -- both entry points below are just views over the same `Agent`.
- `tokio_ai/tui.py` -- the default full-screen terminal UI (`tokio-ai`),
  built with [Textual](https://textual.textualize.io/): a fixed banner,
  scrollable chat log, an input box, and modal screens for chat
  browsing/usage/settings, all dark-themed.
- `tokio_ai/cli.py` -- plain-text REPL fallback (`tokio-ai-plain`), for
  scripting, piping, or terminals that don't support a full-screen TUI.
- `tokio_ai/chat_store.py` -- local chat persistence, one JSON file per
  chat under `~/.tokio_ai/chats/` (no database dependency). Stores the raw
  message history plus the `TestLedger` state, so resuming a chat resumes
  its multiple-testing correction too, not just the transcript.

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

Earlier versions asked the agent to manually crunch raw price history inline
for "does X predict Y"-style questions, which was unreliable on any LLM
backend (not specific to this one) -- that kind of bucketing belongs in a
Python tool, not the model's own token-by-token reasoning over a big JSON
blob. `test_return_pattern` and `top_performing_stocks` now do that
fetch+compute work in Python for the common cases (a technical condition
predicting forward returns; ranking stocks by trailing performance). If you
ask something shaped differently enough that neither tool fits, the model
may still fall back to reasoning over raw data by hand -- treat that path
as unreliable until there's a dedicated tool for it.

### Example

```
> Test whether AAPL days that gap up more than 2% at the open tend to keep
  drifting up over the next 5 trading days, using 10 years of history.

The test found a statistically significant pattern (p=0.0050 after
correction), but in the opposite direction of the initial hypothesis. AAPL
days with gaps >2% at the open saw an average -1.31% return over the next 5
trading days, significantly underperforming the baseline. This suggests
large upward gaps historically preceded short-term weakness (gap-fade), not
momentum continuation. (data window: 2016-08-01 to 2026-07-31)
```

## License

MIT -- see [LICENSE](LICENSE).
