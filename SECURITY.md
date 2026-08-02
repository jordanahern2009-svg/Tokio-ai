# Security

## Reporting a vulnerability

Open a GitHub issue, or if it's sensitive (credential exposure, a way to
make the agent execute arbitrary code), use GitHub's private vulnerability
reporting (Security tab -> "Report a vulnerability") instead of a public
issue.

## What this project touches

- **Your LLM API key** (`OPENAI_API_KEY`), read from a local `.env` file
  and never logged, printed, or sent anywhere except the configured
  `OPENAI_BASE_URL`. Never commit `.env` -- it's gitignored by default.
- **Outbound network calls** to Yahoo Finance, SEC EDGAR, and your
  configured LLM endpoint. No other network access, no telemetry, no
  analytics.
- **Local chat storage** at `~/.tokio_ai/chats/`, one plaintext JSON file
  per chat (your questions, the agent's answers, and the statistical test
  ledger). Never uploaded anywhere -- it's read and written only by your
  own machine. Delete a chat via `Ctrl+P` in the TUI, or just delete the
  files directly; there's no separate "clear history" concept beyond that.
- **No arbitrary code execution.** Tool inputs from the model are typed
  JSON (strings, numbers, arrays) passed to fixed Python functions --
  there is no `eval`, `exec`, or shell-out anywhere in the tool-calling
  path. If you're extending this with a new tool, keep it that way.

## Known limitations, not vulnerabilities

- The agent's data sources (Yahoo, SEC EDGAR) are unauthenticated public
  endpoints; anyone running this tool is trusting those sources the same
  way `curl` would.
- The bundled S&P 500 snapshot (`tokio_ai/data/sp500.csv`) is a point-in-time
  dataset, not a live feed -- see `scripts/refresh_sp500.py`.
