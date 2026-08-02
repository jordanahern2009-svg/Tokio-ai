"""Full-screen terminal UI for TokIO AI, built with Textual.

`tokio-ai` launches this by default. The plain-text REPL (`python -m
tokio_ai.cli`) still exists underneath it for scripting, piping, or
terminals that don't support a full-screen TUI -- this is a presentation
layer over the same `Agent` class, not a second implementation of the loop.
"""

from __future__ import annotations

import asyncio
import os
import sys

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Input, RichLog, Static

from ._env import load_env_file
from .agent.loop import Agent


# Generated once with `pyfiglet.figlet_format("TOKIO AI", font="standard")`
# and hard-coded here rather than adding pyfiglet as a runtime dependency
# for static text. The hand-drawn version this replaced rendered the "I" as
# a plain vertical bar, indistinguishable from a "T" at a glance -- this
# font renders I as a distinct slanted stroke instead.
BANNER = r""" _____ ___  _  _____ ___       _    ___
|_   _/ _ \| |/ /_ _/ _ \     / \  |_ _|
  | || | | | ' / | | | | |   / _ \  | |
  | || |_| | . \ | | |_| |  / ___ \ | |
  |_| \___/|_|\_\___\___/  /_/   \_\___|
"""


class TokioApp(App):
    CSS = """
    Screen {
        background: black;
    }
    #banner {
        content-align: center middle;
        color: #3fb950;
        text-style: bold;
        height: auto;
        padding: 1 0 0 0;
    }
    #tagline {
        content-align: center middle;
        color: #9198a1;
        height: auto;
        padding: 0 0 1 0;
    }
    #chat-log {
        border: round #3fb950;
        background: black;
        margin: 0 2;
    }
    #input-box {
        margin: 1 2;
        border: round #3fb950;
    }
    """
    TITLE = "TokIO AI"
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.agent = Agent()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(BANNER, id="banner")
            yield Static("financial research agent -- ask it anything, it'll test before it claims", id="tagline")
            yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
            yield Input(placeholder="Ask something...", id="input-box")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input-box", Input).focus()
        self.query_one("#chat-log", RichLog).write(
            "[dim]Type a question and press enter. Ctrl+C to quit.[/dim]"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        input_box = self.query_one("#input-box", Input)
        input_box.value = ""
        if not text:
            return
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold #3fb950]> {text}[/bold #3fb950]")
        input_box.disabled = True
        self.run_worker(self._get_reply(text), exclusive=True)

    async def _get_reply(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[dim]thinking...[/dim]")
        try:
            reply = await asyncio.to_thread(self.agent.send, text)
        except Exception as e:
            reply = f"[error] {e}"
        log.write(f"{reply}\n")
        input_box = self.query_one("#input-box", Input)
        input_box.disabled = False
        input_box.focus()


def main() -> None:
    load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Put it in .env or export it before running.", file=sys.stderr)
        sys.exit(1)
    TokioApp().run()


if __name__ == "__main__":
    main()
