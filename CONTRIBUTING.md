# Contributing

Thanks for looking at this. A few things that'll make a PR easy to merge:

## Ground rules

- **No hallucinated data.** Tickers, sector membership, filing data, price
  history -- all of it has to come from a real source (see `tools/`), never
  invented or recalled from a model's memory. This is the one rule that
  isn't negotiable; it's the entire reason this project exists.
- **New tools need tests.** If a function does pure logic (parsing, bucketing,
  statistics), it should have a deterministic unit test with no network
  call -- see `tests/test_patterns.py` or `tests/test_prices.py` for the
  pattern (network-dependent fetch functions are the one thing that's fine
  to leave untested in CI; verify those live instead).
- **Don't add a dependency to save a few lines of stdlib code.** This
  project is deliberately light on dependencies (`openai` is the only
  required one). If you're tempted to reach for a library, check whether
  `urllib`/`statistics`/`concurrent.futures` already covers it.

## Running things locally

```bash
pip install -e ".[dev]"
python -m pytest        # full suite, no network, no API key needed
```

To exercise the live agent loop, copy `.env.example` to `.env` and fill in
a free key from https://build.nvidia.com (or any other OpenAI-compatible
endpoint).

## Adding a new tool

1. Pure logic goes in `tools/your_thing.py`, with the network call (if any)
   as thin as possible so the logic underneath it is testable in isolation.
2. Add a tool schema in `agent/tool_schemas.py` and wire the dispatch in
   `agent/loop.py`.
3. If the tool answers a class of question the model would otherwise
   answer by guessing or hand-crunching raw data, say so explicitly in the
   tool's description and in `agent/system_prompt.py` -- that's usually the
   whole point of adding it.

## Reporting a bug

If you hit a bad result (wrong number, hallucinated fact, tool that should
have fired but didn't), open an issue with the exact prompt you used. This
project cares more about honest failure reports than clean-looking demos.
