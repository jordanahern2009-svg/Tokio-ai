"""Interactive CLI for the TokIO AI agent."""

from __future__ import annotations

import os
import sys

from ._env import load_env_file
from .agent.loop import Agent


def main() -> None:
    load_env_file()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Put it in .env or export it before running.", file=sys.stderr)
        sys.exit(1)

    agent = Agent()
    print("TokIO AI -- financial research agent. Type 'exit' to quit.\n")
    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in ("exit", "quit"):
            break
        if not user_text:
            continue
        try:
            reply = agent.send(user_text)
        except Exception as e:
            print(f"[error] {e}")
            continue
        print(reply)
        print()


if __name__ == "__main__":
    main()
