# Changelog

## 0.3.0 — 2026-08-17

The rigor engine was measured against its own promise for the first time,
failed, and was rewritten. See [docs/calibration.md](docs/calibration.md)
for the full study; `scripts/calibration_study.py` reproduces it with no API
key and no network.

A test that reports `p < 0.05` should be wrong 5% of the time on data with
nothing in it. On simulated price series with no predictable structure
whatsoever, v0.2.0 was wrong up to **45%** of the time (worst case now 7.0%). Two compounding
causes, both specific to what this tool is for:

- **Studentized permutation statistic** (breaking change to p-values).
  `permutation_test` permuted a raw mean difference, which is only valid
  when the two groups are exchangeable. Every condition worth testing here
  ("days that fell 2%", "days with unusual volume") selects volatile days by
  construction, so the condition group is small and much noisier than
  baseline — measured at 1.75x the variance and 15x fewer observations.
  Pooling those makes the small group's mean look far more stable than it
  is. Now permutes a Welch-style studentized statistic, which stays valid
  under unequal variances (Chung & Romano 2013).
- **New `circular_shift_test` for time-ordered data.** Forward returns over
  a multi-day horizon come from overlapping windows and are heavily
  autocorrelated; conditions cluster in time as well. Shuffling labels
  assumes neither is true. Rotating the label series against the value
  series preserves both structures exactly. On a synthetic case with no
  relationship at all, the old approach returned p=0.00033 and the new one
  p=0.49.
- `test_return_pattern` now uses the rotation test. Because only the
  condition-met days move under a rotation, every distinct rotation is
  enumerated rather than sampled — p-values are **exact**, not Monte Carlo,
  and the test suite got ~2.7x faster.
- **The README's own showcase example was a false positive.** "AAPL gaps
  above 2% fade over the next 5 days, p=0.0042" re-runs at **p=0.089** on
  the same real 10-year window. Replaced with the corrected result.
- Verdicts now report which method produced them, and warn explicitly when
  the two groups' variances differ enough that a naive test would have
  overstated the result.
- `PermutationResult` gains `statistic` and `variance_ratio`. Both are
  persisted; older saved chats load unchanged.

Also fixed, unrelated to the above:

- **Chat saves are now atomic.** `save_chat` truncated the real file before
  writing, so a crash or Ctrl+C mid-save destroyed the conversation.
  Writes to a temp file and `os.replace`s it. (`list_chats` already had to
  skip unparseable files — that was the symptom.)
- Loading a chat tolerates unknown fields in a saved result instead of
  raising, so a chat written by a newer TokIO still opens in an older one.
- Persisted result fields derive from the dataclass, so a new field can no
  longer be silently dropped from saved chats.
- Chat listing iterates in sorted order, for stable output.
- Packaging: homepage now points at the docs site, added a Documentation
  URL, Python 3.14 and an Information Analysis classifier.

## 0.2.0 — 2026-08-02

New feature release: a full-screen terminal UI, now the default `tokio-ai`
experience.

- **New `tokio_ai.tui` module**, built with [Textual](https://textual.textualize.io/):
  a dark-themed screen with a "TOKIO AI" banner, a scrollable bordered chat
  log, and an input box pinned at the bottom. Agent calls run in a
  background worker so the UI stays responsive during slower (free-tier)
  LLM round trips.
- `tokio-ai` now launches the TUI by default. The previous plain-text REPL
  moved to `tokio-ai-plain` (still reachable via `python -m tokio_ai.cli`)
  for scripting, piping, or terminals that can't render a full-screen app.
  Pure presentation layer -- both entry points share the same underlying
  `Agent` class, no logic duplicated.
- Verified with Textual's headless pilot test framework (no real terminal
  or network needed) plus a rendered SVG snapshot to visually confirm the
  layout.
- Fixed a real legibility bug caught in review: the initial hand-drawn
  ASCII-art banner rendered the letter "I" as a plain vertical bar,
  indistinguishable from a "T" at a glance. Replaced with a proper figlet
  font where I renders as a distinct slanted stroke.

## 0.1.1 — 2026-08-02

Bug-fix release. A deep-dive audit (manual pass + an independent code-review
pass) found and fixed 7 real bugs, two of which shipped in 0.1.0:

- **`permutation_test` could report `p=0.0` exactly.** With finite Monte
  Carlo resampling that overclaims certainty -- fixed with the standard
  plus-one correction, `(hits+1)/(iters+1)` (Phipson & Smyth 2010).
- **Windows encoding bug, live in 0.1.0**: reading the bundled S&P 500 CSV
  without `encoding="utf-8"` corrupted names like "Brown-Forman" into
  mojibake under Windows' default cp1252 codepage. CI now also runs on
  `windows-latest` (previously Linux-only, which is why this shipped
  unnoticed).
- `TestLedger.verdict()` returned the *first* match by hypothesis name, not
  the latest -- reusing a name silently reported a stale result forever.
- `recent_filings(limit=0)` returned 1 result instead of 0.
- `top_performers(top_n=-N)` silently returned "all but the last N" via
  Python slice semantics instead of erroring.
- Agent conversation history could end up inconsistent after an API
  failure mid-turn, or after exhausting the tool-call round limit.
- `.env` parsing broke on quoted values (`KEY="value"`) and had the same
  missing-encoding bug as the CSV reader.

Also: `top_performing_stocks` now surfaces an explicit warning when data
yield drops below 90%, instead of relying on the caller to notice by
comparing two numbers. Test suite grew from 44 to 57 tests, all with
regression coverage for the bugs above.

## 0.1.0 — 2026-08-02

Initial release: price history and SEC filings lookups, a real S&P 500
screener with GICS sector filtering, technical-pattern hypothesis testing,
and the core rigor engine (permutation testing + Bonferroni/
Benjamini-Hochberg multiple-testing correction). Agent runs against any
OpenAI-compatible endpoint, defaulting to NVIDIA's free NIM catalog.
