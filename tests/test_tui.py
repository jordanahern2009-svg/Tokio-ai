"""Headless test of the Textual TUI using its built-in pilot framework --
no real terminal, no network, no API key needed. Wrapped in a plain sync
test via asyncio.run() rather than pulling in pytest-asyncio as a new
dependency for one test file."""

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "dummy-test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid/v1")

from tokio_ai import chat_store
from tokio_ai.tui import SPINNER_FRAMES, TokioApp

_ORIGINAL_STORE_DIR = chat_store.STORE_DIR


def _fake_response(content: str):
    # Mocking at this level (the HTTP client) rather than replacing
    # agent.send directly means the real Agent.send() logic still runs --
    # including appending to agent.messages, which chat_store persistence
    # depends on. Mocking agent.send itself would silently skip that.
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=None,
    )


def setup_module(module) -> None:
    # Every test in this file touches chat_store (TokioApp saves/loads on
    # construction and after every reply) -- redirect it to a throwaway
    # directory for the whole file rather than the real ~/.tokio_ai.
    chat_store.STORE_DIR = Path(tempfile.mkdtemp())


def teardown_module(module) -> None:
    shutil.rmtree(chat_store.STORE_DIR, ignore_errors=True)
    chat_store.STORE_DIR = _ORIGINAL_STORE_DIR


def _new_app() -> TokioApp:
    # Each test gets its own fresh sub-directory so tests can't see each
    # other's saved chats (e.g. "no chats yet" tests would break otherwise).
    chat_store.STORE_DIR = Path(tempfile.mkdtemp())
    return TokioApp()


def test_tui_renders_banner_and_handles_a_chat_turn():
    async def scenario():
        app = _new_app()
        app.agent.client.chat.completions.create = lambda **kwargs: _fake_response("MOCKED REPLY")

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            log = app.query_one("#chat-log")
            initial_lines = len(log.lines)
            assert initial_lines >= 1  # the startup hint line

            await pilot.click("#input-box")
            for ch in "hello":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause(delay=0.3)

            rendered = "\n".join(str(line) for line in log.lines)
            assert "> hello" in rendered
            assert "MOCKED REPLY" in rendered

            input_box = app.query_one("#input-box")
            assert input_box.value == ""  # cleared after submit
            assert not input_box.disabled  # re-enabled after the reply comes back

            # also persisted to disk
            saved_messages, _, _ = chat_store.load_chat(app.current_chat_id)
            assert any(m.get("content") == "hello" for m in saved_messages)

    asyncio.run(scenario())


def test_tui_shows_animated_status_while_waiting_and_clears_it_after():
    # Real feedback from actually trying this: free-tier responses can take
    # 5-90s+, and a static "thinking..." with no movement reads as frozen
    # on the slow end. Uses a real (blocking, via asyncio.to_thread) delay
    # rather than mocking time, so this exercises the actual timer.
    async def scenario():
        app = _new_app()
        app.agent.send = lambda text: (time.sleep(0.6), "done")[1]

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            await pilot.click("#input-box")
            for ch in "hi":
                await pilot.press(ch)
            await pilot.press("enter")

            await pilot.pause(delay=0.15)
            status = app.query_one("#status")
            first = str(status.render())
            assert "thinking" in first
            assert any(frame in first for frame in SPINNER_FRAMES)

            await pilot.pause(delay=0.3)
            second = str(status.render())
            assert first != second  # the spinner frame/elapsed time actually advanced

            await pilot.pause(delay=0.5)  # let the 0.6s reply finish
            assert str(app.query_one("#status").render()) == ""  # cleared once the reply lands

    asyncio.run(scenario())


def test_tui_ignores_empty_submission():
    async def scenario():
        app = _new_app()
        app.agent.send = lambda text: "should not be called"

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            log = app.query_one("#chat-log")
            before = len(log.lines)

            await pilot.click("#input-box")
            await pilot.press("enter")  # submit with empty input
            await pilot.pause()

            assert len(log.lines) == before  # nothing was logged

    asyncio.run(scenario())


def test_new_chat_clears_log_and_starts_a_new_id():
    async def scenario():
        app = _new_app()
        app.agent.client.chat.completions.create = lambda **kwargs: _fake_response("reply one")

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            first_chat_id = app.current_chat_id

            await pilot.click("#input-box")
            for ch in "first message":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause(delay=0.3)

            app.action_new_chat()
            await pilot.pause()

            assert app.current_chat_id != first_chat_id
            rendered = "\n".join(str(line) for line in app.query_one("#chat-log").lines)
            assert "first message" not in rendered
            assert len(app.agent.messages) == 1  # system prompt only

            # the old chat was actually saved, not lost
            saved_messages, _, _ = chat_store.load_chat(first_chat_id)
            assert any(m.get("content") == "first message" for m in saved_messages)

    asyncio.run(scenario())


def test_browsing_and_switching_chats_loads_the_right_history():
    async def scenario():
        app = _new_app()
        app.agent.client.chat.completions.create = lambda **kwargs: _fake_response("some reply")

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            await pilot.click("#input-box")
            for ch in "chat A message":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause(delay=0.3)
            chat_a_id = app.current_chat_id

            app.action_new_chat()
            await pilot.pause()
            await pilot.click("#input-box")
            for ch in "chat B message":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause(delay=0.3)

            # Switch back to chat A directly via the same handler the
            # ChatListScreen callback would invoke, rather than driving the
            # modal's ListView pixel-by-pixel.
            messages, ledger, usage = chat_store.load_chat(chat_a_id)
            app.current_chat_id = chat_a_id
            app.agent.messages = messages
            app.agent.ledger = ledger
            app.agent.usage.update(usage or {})
            app._render_history()
            await pilot.pause()

            rendered = "\n".join(str(line) for line in app.query_one("#chat-log").lines)
            assert "chat A message" in rendered
            assert "chat B message" not in rendered

    asyncio.run(scenario())


def test_usage_screen_shows_real_accumulated_numbers():
    async def scenario():
        app = _new_app()
        app.agent.usage.update({"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50, "request_count": 2})

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.action_show_usage()
            await pilot.pause()

            screen = app.screen
            text_widgets = [w for w in screen.query("Static")]
            combined = " ".join(str(w.render()) for w in text_widgets)
            assert "42" in combined
            assert "8" in combined
            assert "50" in combined

    asyncio.run(scenario())


def test_settings_screen_updates_model_and_permission():
    async def scenario():
        app = _new_app()

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            original_model = app.agent.model
            assert app.agent.permission_level == "auto"

            app.action_show_settings()
            await pilot.pause()

            model_input = app.screen.query_one("#model-input")
            model_input.value = "some/other-model"
            permission_select = app.screen.query_one("#permission-select")
            permission_select.value = "confirm"

            await pilot.click("#save-btn")
            await pilot.pause()

            assert app.agent.model == "some/other-model"
            assert app.agent.permission_level == "confirm"
            assert app.agent.model != original_model

    asyncio.run(scenario())


def test_settings_screen_cancel_does_not_change_anything():
    async def scenario():
        app = _new_app()

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            original_model = app.agent.model
            original_permission = app.agent.permission_level

            app.action_show_settings()
            await pilot.pause()
            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.agent.model == original_model
            assert app.agent.permission_level == original_permission

    asyncio.run(scenario())


def test_permission_confirm_flow_denies_and_still_completes():
    async def scenario():
        app = _new_app()
        app.agent.permission_level = "confirm"

        def fake_send(text):
            # Simulate the agent calling the confirm hook mid-turn, exactly
            # as _execute_tool would for a real tool call.
            approved = app.agent.confirm_callback("get_price_history", {"symbol": "AAPL"})
            return "approved!" if approved else "denied, moving on"

        app.agent.send = fake_send

        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            await pilot.click("#input-box")
            for ch in "get AAPL data":
                await pilot.press(ch)
            await pilot.press("enter")

            # Wait for the confirmation prompt to appear (it's produced from
            # a background thread via call_from_thread).
            for _ in range(50):
                await pilot.pause(delay=0.05)
                rendered = "\n".join(str(line) for line in app.query_one("#chat-log").lines)
                if "Confirm:" in rendered:
                    break
            assert "Confirm:" in rendered
            assert not app.query_one("#input-box").disabled  # re-enabled so you can answer

            await pilot.click("#input-box")
            await pilot.press("n")
            await pilot.press("enter")

            for _ in range(50):
                await pilot.pause(delay=0.05)
                rendered = "\n".join(str(line) for line in app.query_one("#chat-log").lines)
                if "denied, moving on" in rendered:
                    break
            assert "denied, moving on" in rendered

    asyncio.run(scenario())
