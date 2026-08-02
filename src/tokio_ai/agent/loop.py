"""Tool-use loop against any OpenAI-compatible chat-completions endpoint.

Defaults to NVIDIA's free NIM catalog (integrate.api.nvidia.com) rather than
a paid API, on purpose: an open-source tool that requires a metered key to
even try is a real adoption barrier. Point OPENAI_BASE_URL/OPENAI_API_KEY at
any other OpenAI-compatible endpoint (OpenRouter, a local vLLM/Ollama
server, etc.) and this keeps working unmodified.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from ..rigor.ledger import TestLedger
from ..tools.filings import recent_filings
from ..tools.hypothesis import test_hypothesis as _test_hypothesis
from ..tools.prices import fetch_daily_bars
from .system_prompt import SYSTEM_PROMPT
from .tool_schemas import TOOLS, to_openai_format

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Picked empirically, not from a spec sheet: of the 4 candidates tried on
# NVIDIA's free tier, this was the only one that (a) was actually enabled for
# a free account and (b) reliably returned well-formed tool calls in well
# under the timeout. meta/llama-3.3-70b-instruct consistently timed out
# (>90s, likely not warm on the free tier); writer/palmyra-fin-70b-32k and
# mistralai/mistral-large-2-instruct both 404'd as not enabled for this
# account tier. Nemotron is also NVIDIA's own agentic/tool-use-tuned model,
# which tracks with it being the most reliable one here.
DEFAULT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_TIMEOUT = 60.0  # reasoning models can take a while; don't time out mid-thought
MAX_TOOL_ROUNDS = 8  # hard cap so a confused loop can't spin forever

OPENAI_TOOLS = to_openai_format(TOOLS)


class Agent:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            timeout=DEFAULT_TIMEOUT,
            max_retries=1,
        )
        self.model = model or os.environ.get("TOKIO_AI_MODEL", DEFAULT_MODEL)
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.ledger = TestLedger()

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        try:
            if name == "get_price_history":
                bars = fetch_daily_bars(tool_input["symbol"], tool_input.get("range", "10y"))
                return json.dumps([b.to_dict() for b in bars[-500:]])  # cap payload size
            if name == "get_sec_filings":
                filings = recent_filings(
                    tool_input["symbol"], tool_input.get("form_type"), tool_input.get("limit", 10)
                )
                return json.dumps([f.to_dict() for f in filings])
            if name == "test_hypothesis":
                return _test_hypothesis(
                    self.ledger, tool_input["name"], tool_input["group_a"], tool_input["group_b"]
                )
            return f"ERROR: unknown tool {name!r}"
        except Exception as e:  # tool errors go back to the model as text, never crash the loop
            return f"ERROR: {e}"

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                messages=self.messages,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content or ""})
                return msg.content or ""

            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                result_text = self._execute_tool(tc.function.name, tool_input)
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                )

        return "[stopped: too many tool calls in a row without a final answer]"
