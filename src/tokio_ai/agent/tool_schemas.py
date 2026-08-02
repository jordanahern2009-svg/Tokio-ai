def to_openai_format(tools: list[dict]) -> list[dict]:
    """Adapt this project's tool defs (name/description/input_schema) to the
    OpenAI-compatible function-calling shape used by NVIDIA NIM and friends."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


TOOLS = [
    {
        "name": "get_price_history",
        "description": (
            "Fetch daily OHLCV price history for a stock ticker from Yahoo "
            "Finance. No API key needed. Returns a list of daily bars with "
            "date/open/high/low/close/adj_close/volume."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
                "range": {
                    "type": "string",
                    "description": "History window: 1y, 5y, 10y, 20y. Avoid requesting more than 20y.",
                    "default": "10y",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_sec_filings",
        "description": (
            "List recent SEC filings for a ticker (e.g. 10-K, 10-Q, 8-K) via "
            "EDGAR. No API key needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "form_type": {
                    "type": "string",
                    "description": "e.g. '10-K', '10-Q', '8-K'. Omit for all types.",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "test_hypothesis",
        "description": (
            "Run a rigorous two-sided permutation test comparing two groups of "
            "numbers (e.g. forward returns after a signal fires vs. a "
            "baseline). ALWAYS use this before claiming any pattern is real -- "
            "never eyeball a mean and call it significant. Enforces a minimum "
            "sample size and corrects for every other hypothesis tested this "
            "session, so results only get more conservative as more signals "
            "get tested in one conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short unique label for this hypothesis, e.g. 'AAPL_positive_surprise_20d'",
                },
                "group_a": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "e.g. forward returns in the 'signal fired' condition",
                },
                "group_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "e.g. forward returns in the baseline/control condition",
                },
            },
            "required": ["name", "group_a", "group_b"],
        },
    },
]
