"""Full-screen terminal UI for TokIO AI, built with Textual.

`tokio-ai` launches this by default. The plain-text REPL (`python -m
tokio_ai.cli`) still exists underneath it for scripting, piping, or
terminals that don't support a full-screen TUI -- this is a presentation
layer over the same `Agent` class, not a second implementation of the loop.

Supports multiple named, disk-persisted chats (ctrl+n new, ctrl+p browse),
a usage view (ctrl+u), and a settings screen for model + tool-permission
level (ctrl+o). All state lives in Agent/chat_store; this module only
renders it and wires input.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Footer, Input, Label, ListItem, ListView, RichLog, Select, Static

from . import chat_store
from ._env import load_env_file
from .agent.loop import Agent, PERMISSION_LEVELS
from .rigor.ledger import TestLedger

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

DIALOG_CSS = """
    align: center middle;
"""

DIALOG_PANEL_CSS = """
    width: 80%;
    height: 80%;
    border: round #3fb950;
    background: black;
    padding: 1 2;
"""


class ChatListScreen(ModalScreen[str | None]):
    """Pick a past chat to resume. Returns the chat id, or None on cancel."""

    CSS = f"""
    ChatListScreen {{ {DIALOG_CSS} }}
    #dialog {{ {DIALOG_PANEL_CSS} }}
    #dialog-title {{ color: #3fb950; text-style: bold; height: auto; padding-bottom: 1; }}
    ListView {{ background: black; }}
    """
    BINDINGS = [("escape", "cancel", "Cancel"), ("d", "delete_selected", "Delete")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Chats -- enter to open, d to delete, esc to cancel", id="dialog-title")
            yield ListView(id="chat-list")

    def on_mount(self) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        list_view = self.query_one("#chat-list", ListView)
        list_view.clear()
        self._ids: list[str | None] = []
        summaries = chat_store.list_chats()
        if not summaries:
            list_view.append(ListItem(Label("No saved chats yet.")))
            self._ids.append(None)
            return
        for s in summaries:
            when = s.updated_at[:16].replace("T", " ") if s.updated_at else "?"
            list_view.append(ListItem(Label(f"{s.title}  [dim]({s.message_count} msgs, {when})[/dim]")))
            self._ids.append(s.id)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_delete_selected(self) -> None:
        list_view = self.query_one("#chat-list", ListView)
        index = list_view.index
        if index is None or index >= len(self._ids) or self._ids[index] is None:
            return
        chat_store.delete_chat(self._ids[index])
        self._refresh_list()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        list_view = self.query_one("#chat-list", ListView)
        index = list_view.index
        chat_id = self._ids[index] if index is not None and index < len(self._ids) else None
        self.dismiss(chat_id)


class UsageScreen(ModalScreen[None]):
    """Read-only view of this chat's token usage. Not a dollar cost --
    NVIDIA's free tier has none; this is what it actually is, token/request
    counts from the real API responses."""

    CSS = f"""
    UsageScreen {{ {DIALOG_CSS} }}
    #dialog {{ {DIALOG_PANEL_CSS} width: 50%; height: auto; }}
    #dialog-title {{ color: #3fb950; text-style: bold; height: auto; padding-bottom: 1; }}
    """
    BINDINGS = [("escape", "dismiss_self", "Close")]

    def __init__(self, usage: dict, model: str) -> None:
        super().__init__()
        self._usage = usage
        self._model = model

    def compose(self) -> ComposeResult:
        u = self._usage
        text = (
            f"Model: {self._model}\n\n"
            f"Requests this chat: {u['request_count']}\n"
            f"Prompt tokens:      {u['prompt_tokens']:,}\n"
            f"Completion tokens:  {u['completion_tokens']:,}\n"
            f"Total tokens:       {u['total_tokens']:,}\n\n"
            "[dim]This is token/request counts from real API responses, not a\n"
            "dollar cost -- NVIDIA's free NIM tier has none. If you point this\n"
            "at a paid endpoint, check that provider's own billing dashboard\n"
            "for actual cost.[/dim]"
        )
        with Vertical(id="dialog"):
            yield Static("Usage (this chat)", id="dialog-title")
            yield Static(text)

    def action_dismiss_self(self) -> None:
        self.dismiss(None)


class SettingsScreen(ModalScreen[tuple[str, str] | None]):
    """Model + tool-permission settings. Returns (model, permission_level)
    on save, or None on cancel."""

    CSS = f"""
    SettingsScreen {{ {DIALOG_CSS} }}
    #dialog {{ {DIALOG_PANEL_CSS} width: 60%; height: auto; }}
    #dialog-title {{ color: #3fb950; text-style: bold; height: auto; padding-bottom: 1; }}
    Label {{ padding-top: 1; }}
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, current_model: str, current_permission: str) -> None:
        super().__init__()
        self._current_model = current_model
        self._current_permission = current_permission

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Settings", id="dialog-title")
            yield Label(
                "Model (the default has been verified to support tool-calling on the "
                "free tier; other model IDs may not -- your account's access varies "
                "by model, this isn't checked for you):"
            )
            yield Input(value=self._current_model, id="model-input")
            yield Label("Tool permissions:")
            yield Select(
                [("Auto-approve (recommended)", "auto"), ("Confirm before each tool call", "confirm")],
                value=self._current_permission,
                allow_blank=False,
                id="permission-select",
            )
            with Horizontal():
                yield Button("Save", id="save-btn", variant="success")
                yield Button("Cancel", id="cancel-btn")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            model = self.query_one("#model-input", Input).value.strip()
            permission = self.query_one("#permission-select", Select).value
            self.dismiss((model, permission))
        else:
            self.dismiss(None)


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
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+n", "new_chat", "New chat"),
        ("ctrl+p", "browse_chats", "Chats"),
        ("ctrl+u", "show_usage", "Usage"),
        ("ctrl+o", "show_settings", "Settings"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent = Agent()
        self.agent.confirm_callback = self._confirm_tool_call

        existing = chat_store.list_chats()
        if existing:
            self.current_chat_id = existing[0].id
            messages, ledger, usage = chat_store.load_chat(self.current_chat_id)
            self.agent.messages = messages
            self.agent.ledger = ledger
            self.agent.usage.update(usage or {})
        else:
            self.current_chat_id = chat_store.new_chat_id()

        self._spinner_timer: Timer | None = None
        self._spinner_index = 0
        self._think_started_at = 0.0
        self._pending_confirm: threading.Event | None = None
        self._confirm_answer = False

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
        self._render_history()

    def _render_history(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        shown_any = False
        for m in self.agent.messages:
            role = m.get("role")
            content = m.get("content")
            if role == "user" and content:
                log.write(f"[bold #3fb950]> {content}[/bold #3fb950]")
                shown_any = True
            elif role == "assistant" and content:
                log.write(f"{content}\n")
                shown_any = True
        if not shown_any:
            log.write("[dim]Type a question and press enter. Ctrl+C to quit, "
                       "Ctrl+N new chat, Ctrl+P browse chats, Ctrl+U usage, Ctrl+O settings.[/dim]")

    def _save_current_chat(self) -> None:
        chat_store.save_chat(self.current_chat_id, self.agent.messages, self.agent.ledger, self.agent.usage)

    # -- chat management -----------------------------------------------

    def action_new_chat(self) -> None:
        if len(self.agent.messages) > 1:  # more than just the system prompt
            self._save_current_chat()
        self.current_chat_id = chat_store.new_chat_id()
        self.agent.messages = self.agent.messages[:1]  # keep the system prompt only
        self.agent.ledger = TestLedger()
        self._render_history()

    def action_browse_chats(self) -> None:
        if len(self.agent.messages) > 1:
            self._save_current_chat()

        def handle_choice(chat_id: str | None) -> None:
            if chat_id is None or chat_id == self.current_chat_id:
                return
            self.current_chat_id = chat_id
            messages, ledger, usage = chat_store.load_chat(chat_id)
            self.agent.messages = messages
            self.agent.ledger = ledger
            self.agent.usage.update(usage or {})
            self._render_history()

        self.push_screen(ChatListScreen(), handle_choice)

    def action_show_usage(self) -> None:
        self.push_screen(UsageScreen(dict(self.agent.usage), self.agent.model))

    def action_show_settings(self) -> None:
        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            model, permission = result
            if model:
                self.agent.model = model
            if permission in PERMISSION_LEVELS:
                self.agent.permission_level = permission

        self.push_screen(SettingsScreen(self.agent.model, self.agent.permission_level), handle_result)

    # -- tool-call confirmation (permission_level == "confirm") --------

    def _confirm_tool_call(self, name: str, tool_input: dict) -> bool:
        """Called from the background worker thread via Agent.confirm_callback
        -- blocks that thread (not the UI thread) until the user answers."""
        event = threading.Event()
        self._pending_confirm = event
        self._confirm_answer = False

        def show_prompt() -> None:
            self.query_one("#chat-log", RichLog).write(
                f"[#ffa62b]Confirm: run {name}({tool_input})? Type y/n and press enter.[/#ffa62b]"
            )
            input_box = self.query_one("#input-box", Input)
            input_box.disabled = False
            input_box.focus()

        self.call_from_thread(show_prompt)
        event.wait()
        return self._confirm_answer

    # -- chat input ------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        input_box = self.query_one("#input-box", Input)
        input_box.value = ""

        if self._pending_confirm is not None:
            confirm_event = self._pending_confirm
            self._pending_confirm = None
            self._confirm_answer = text.lower() in ("y", "yes")
            log = self.query_one("#chat-log", RichLog)
            log.write(f"[dim]-> {'approved' if self._confirm_answer else 'denied'}[/dim]")
            input_box.disabled = True  # agent is still working on the rest of the turn
            confirm_event.set()
            return

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
        self._save_current_chat()
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
