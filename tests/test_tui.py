"""Headless test of the Textual TUI using its built-in pilot framework --
no real terminal, no network, no API key needed. Wrapped in a plain sync
test via asyncio.run() rather than pulling in pytest-asyncio as a new
dependency for one test file."""

import asyncio
import os

os.environ.setdefault("OPENAI_API_KEY", "dummy-test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid/v1")

from tokio_ai.tui import TokioApp


def test_tui_renders_banner_and_handles_a_chat_turn():
    async def scenario():
        app = TokioApp()
        app.agent.send = lambda text: f"MOCKED REPLY to: {text}"

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
            assert "MOCKED REPLY to: hello" in rendered

            input_box = app.query_one("#input-box")
            assert input_box.value == ""  # cleared after submit
            assert not input_box.disabled  # re-enabled after the reply comes back

    asyncio.run(scenario())


def test_tui_ignores_empty_submission():
    async def scenario():
        app = TokioApp()
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
