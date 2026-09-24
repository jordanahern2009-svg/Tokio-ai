# TokIO AI

[![tests](https://github.com/jordanahern2009-svg/Tokio-ai/actions/workflows/test.yml/badge.svg)](https://github.com/jordanahern2009-svg/Tokio-ai/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/tokio-ai.svg)](https://pypi.org/project/tokio-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![calibrated](https://img.shields.io/badge/false%20positive%20rate-measured-brightgreen.svg)](docs/calibration.md)

An open-source financial research agent that treats "the data supports this"
as a claim to be tested, not a vibe to be trusted.

Give it a ticker, a filing, or a plain-English trading hypothesis. It pulls
real data (price history, SEC filings) and, before it will tell you a
pattern is real, it runs the comparison through a permutation test, checks
the sample size against a hard floor, and corrects for every other
hypothesis you've asked it to test in the same conversation. Most AI
stock-chat tools will confidently describe a pattern in a handful of data
points. This one is built to tell you when it can't.

## We tested the tests, and they failed

Most tools that promise statistical rigor never check whether their own
statistics work. We checked ours, by running it on thousands of simulated
price series containing **no predictable pattern at all** and counting how
often it claimed to find one. A test that reports `p < 0.05` should be wrong
about 5% of the time.

TokIO v0.2.0 was wrong up to **45%** of the time.

Two compounding bugs, both specific to the questions this tool exists to
answer. Conditions like "days that dropped more than 2%" select volatile
days *by construction*, and a permutation test on a raw mean difference
isn't valid when one group is far noisier than the other. On top of that,
multi-day forward returns come from overlapping windows, so shuffling them
pretends there is far more independent data than there is.

v0.3.0 fixes both — a studentized statistic (Chung & Romano 2013) and a
circular-shift randomization that preserves the time structure instead of
destroying it. Worst-case false-positive rate went from **45% to 7%**, and the
engine got about 7x faster, because rotations are cheap enough to
enumerate exactly rather than sample.

The most useful thing the study found was in our own README. The showcase
example here used to be "AAPL gaps above 2% fade over the next 5 days,
p = 0.0042." Re-run with the corrected test on the same 10 years of real
data: **p = 0.089.** Not significant. The headline result was a false
positive produced by the bug.

**[Read the full study →](docs/calibration.md)** — or reproduce it yourself,
no API key or network needed:

```bash
python scripts/calibration_study.py
```

## Check your own backtest

No agent, no API key, no network. You bring returns and a condition you
think predicts them; `check()` tells you whether that's distinguishable
from noise.

```python
import tokio_ai

r = prices.pct_change()                     # your data: list, numpy or pandas
past = prices.shift(20)
momentum = (prices > past).where(past.notna())  # anything known at the bar's close
print(tokio_ai.check(r, momentum, horizon=20))
```

(The `.where(...)` is there because pandas evaluates `NaN > x` as `False`,
not NaN. Without it, the first 20 bars, where momentum is unknown, get
counted as "momentum down". `check()` skips None and NaN, but it can't
recover a NaN that pandas has already turned into `False`.)

Here is that exact check on 10 years of real SPY closes (2016-09 to 2026-09):

```
NOT SIGNIFICANT (p=0.3274, alpha=0.05). Over the next 20 bars, the 1742
condition bars averaged +1.003% vs +1.890% on the other 731 (gap -0.887%).
```

(The rotation engine, as a second opinion: p = 0.31.)

A Welch t-test on the same two groups returns **p = 0.0001**. That's how
you end up "discovering" a mean-reversion edge. The trap is that
consecutive 20-day returns share 19 of their 20 days, so 2,500 bars carry
nowhere near 2,500 independent observations. The t-test doesn't know that.

What `check()` handles for you:

- **Lookahead.** The outcome for bar *i* starts at bar *i+1*, so a
  condition can never predict its own bar. Two pandas Series with
  different indexes raise an error instead of being silently paired by
  position.
- **Overlapping windows and persistent conditions**, handled exactly by
  the default Hodrick engine, and checked by a rotation test that keeps
  the time structure of both series intact.
- **Volatility-selecting conditions**, via heteroskedasticity-robust
  statistics, plus a note telling you which way a naive test would have
  been wrong.
- **Testing many ideas.** Pass `ledger=tokio_ai.TestLedger()` to every
  call, and each verdict is Benjamini-Hochberg corrected against all of
  them.
- **Tiny samples.** Fewer than 30 bars on either side returns
  `NOT REPORTABLE`, not a p-value.

**Two engines on every call.** The verdict comes from Hodrick (1992)
standard errors, which handle the overlap between multi-bar windows
*exactly* instead of estimating it, plus a short HAC for the returns' own
autocorrelation. A circular-shift randomization test runs alongside as an
assumption-free second opinion, and the result tells you when they
disagree. With numpy installed both are fast: the rotation test evaluates
every rotation through an FFT, in 1.2 s at a million bars.

**How it compares to the tools quants already use**, on 51 simulated
markets with no edge (size) and with a planted one (power):

| test | worst false-positive rate | mean power |
|---|---:|---:|
| Welch t-test | 64.0% | 74.0% |
| Newey-West (HAC, lags = h) | 15.0% | 74.2% |
| stationary bootstrap (`arch`) | 15.0% | 74.3% |
| **TokIO `check()`** | **7.7%** | 73.7% |

Same power as Newey-West, within half a point, at half its worst-case
false-positive rate. The default engine was picked by that benchmark,
after two other designs lost. [The full head-to-head, what we tried, and
where each engine is weaker
→](docs/calibration.md#head-to-head-tokio-vs-newey-west-vs-the-stationary-bootstrap)

## Why this exists

Generic LLM agents are commoditized -- anyone can wrap an LLM in a chat loop
and call it an agent. What isn't commoditized is discipline: most retail
(and plenty of professional) research fails because someone eyeballs a mean,
sees a gap, and calls it an edge without asking how likely that gap was to
appear by chance. The rigor layer here (`tokio_ai.rigor`) generalizes a
hypothesis-testing discipline actually used across real trading research
projects -- see `rigor/stats.py` and `rigor/ledger.py` for the
randomization tests, minimum-sample gate, and Bonferroni/Benjamini-Hochberg
multiple-testing correction that every claim has to pass through.

Discipline you can't verify is just a claim, though, which is why
[`docs/calibration.md`](docs/calibration.md) measures whether any of it
actually holds -- and reports where it didn't.

## What it can do today (v0)

- Pull daily OHLCV price history for any ticker (Yahoo Finance, no key)
- Pull recent SEC filings for any ticker (EDGAR, no key)
- Rank the real S&P 500 by trailing return, with optional GICS sector
  filtering -- handles open-ended asks like "what are the best performing
  stocks" without requiring you to already know a ticker or sector
- Test whether a simple technical condition (a big daily move, a gap at the
  open, unusual volume) actually predicts what happens next -- fetches,
  buckets, and runs the test in one call, not via the model eyeballing
  raw numbers -- using a circular-shift randomization that accounts for
  overlapping forward windows
- Run a two-sided permutation test comparing any two groups of numbers you
  already have -- studentized, so it stays honest when one group is much
  noisier than the other -- with automatic multiple-testing correction
  across everything tested in the session
- Chat with it via a full-screen terminal UI with a persistent sidebar (new chat, chat list, usage, settings), or a plain-text REPL (`tokio-ai-plain`); it decides when to call which tool
- Multiple named, disk-persisted chats -- start a new one from the sidebar, click back into old ones, each keeps its own multiple-testing correction history so resuming picks up exactly where you left off
- A usage view showing real token/request counts for the current chat (not a dollar cost -- the free tier has none)
- A settings screen for the model (free-text override; only the default is verified to support tool-calling) and tool-call permissions (auto-approve, or confirm every call)

## TUI layout

Sidebar on the left (new chat, chat list, usage/settings buttons), chat on
the right -- click-driven like a normal chat app, not a keybindings-only
REPL. A few shortcuts still exist for the same actions:

| Key | Action |
|---|---|
| `Ctrl+N` | New chat |
| `Ctrl+U` | Usage (tokens, requests, this chat) |
| `Ctrl+O` | Settings (model, tool permissions) |
| `Ctrl+C` | Quit |

Textual's built-in command palette is intentionally disabled -- it defaults
to `Ctrl+P`, which collided with this app's own bindings, and its "change
theme" command has no visible effect here since the CSS uses fixed colors
rather than Textual's theme-variable system (a real, working theme switcher
is planned, not built yet).

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

- `tokio_ai/rigor/` -- pure-Python statistics engine (studentized
  permutation testing, circular-shift randomization for time-ordered data,
  multiple-testing correction, session-level test ledger). Fully unit
  tested, zero dependencies beyond the standard library, and its
  false-positive rate is measured rather than assumed
  ([`docs/calibration.md`](docs/calibration.md), reproducible via
  `scripts/calibration_study.py`).
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
end-to-end against NVIDIA's free NIM catalog (`nvidia/nemotron-3-super-120b-a12b`
by default) -- real tool calls, real data, correct multi-turn answers.
Free-tier models do get retired: the previous default went dark on
2026-08-26. If you get an HTTP 410 "end of life" error, set `TOKIO_AI_MODEL`
to another tool-calling model from https://build.nvidia.com.

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

Not significant (p=0.089). Over 2016-08-18 to 2026-08-17 there were 76 days
with a gap above 2%, and they averaged -1.29% over the next 5 trading days
versus baseline -- so the point estimate does lean toward gap-fade rather
than continuation, but not by enough to separate from chance.

Worth flagging: those 76 days have 2.68x the return variance of the other
2,430. That is the condition selecting volatile days, and it is exactly the
case where a naive test overstates significance -- this one reported
p=0.0042 on the same data before v0.3.0.
```

That second paragraph is the whole point of the project. The interesting
answer was not the pattern; it was the reason to distrust the pattern.

## License

MIT -- see [LICENSE](LICENSE).
