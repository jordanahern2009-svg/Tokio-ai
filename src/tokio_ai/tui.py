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
import time

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import Footer, Input, RichLog, Static

from ._env import load_env_file
from .agent.loop import Agent

# Braille spinner frames -- standard terminal-spinner glyph set, distinct
# enough from any letterform to never be mistaken for banner text.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


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
    #status {
        color: #9198a1;
        height: 1;
        margin: 0 3;
    }
    #input-box {
        margin: 0 2 1 2;
        border: round #3fb950;
    }
    """
    TITLE = "TokIO AI"
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.agent = Agent()
        self._spinner_timer: Timer | None = None
        self._spinner_index = 0
        self._think_started_at = 0.0

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(BANNER, id="banner")
            yield Static("financial research agent -- ask it anything, it'll test before it claims", id="tagline")
            yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
            yield Static("", id="status")
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
        self._start_thinking()
        self.run_worker(self._get_reply(text), exclusive=True)

    def _start_thinking(self) -> None:
        # Free-tier responses can take anywhere from ~5s to 90s+ -- a static
        # "thinking..." with no movement reads as frozen on the slow end.
        # This lives in its own status line, not the log, since RichLog
        # entries are append-only and can't be updated in place.
        self._spinner_index = 0
        self._think_started_at = time.monotonic()
        self._tick_spinner()
        self._spinner_timer = self.set_interval(0.1, self._tick_spinner)

    def _tick_spinner(self) -> None:
        frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
        self._spinner_index += 1
        elapsed = time.monotonic() - self._think_started_at
        self.query_one("#status", Static).update(f"{frame} thinking... {elapsed:.0f}s")

    def _stop_thinking(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        self.query_one("#status", Static).update("")

    async def _get_reply(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        try:
            reply = await asyncio.to_thread(self.agent.send, text)
        except Exception as e:
            reply = f"[error] {e}"
        self._stop_thinking()
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
