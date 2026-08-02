# Changelog

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
