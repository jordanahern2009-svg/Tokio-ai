"""Tests for the agent tool-use loop, using a mocked OpenAI-compatible
client so nothing here touches the network or needs a real API key."""

from types import SimpleNamespace

from tokio_ai.agent.loop import Agent, MAX_TOOL_ROUNDS


def _fake_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15):
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)


def _fake_response(content=None, tool_calls=None, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=usage,
    )


def _fake_tool_call(id_: str, name: str, arguments: str):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def _new_agent() -> Agent:
    return Agent(api_key="dummy-test-key", base_url="https://example.invalid/v1")


def test_send_rolls_back_history_on_api_failure():
    # Real bug found in manual testing: a failed API call left the user's
    # message dangling in history, so the next send() would stack a second
    # user message right after it instead of retrying cleanly.
    agent = _new_agent()
    before = list(agent.messages)

    def raise_error(**kwargs):
        raise RuntimeError("simulated network failure")

    agent.client.chat.completions.create = raise_error
    try:
        agent.send("hello")
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass
    assert agent.messages == before


def test_send_returns_final_text_and_records_it():
    agent = _new_agent()
    agent.client.chat.completions.create = lambda **kwargs: _fake_response(content="the answer")
    reply = agent.send("a question")
    assert reply == "the answer"
    assert agent.messages[-1] == {"role": "assistant", "content": "the answer"}


def test_send_records_bailout_message_when_max_rounds_exhausted():
    # Real bug found in manual testing: hitting MAX_TOOL_ROUNDS returned the
    # bailout string to the caller but never appended it to self.messages,
    # leaving history ending mid-tool-exchange (a "tool" message with no
    # assistant response), an invalid shape for the next turn's API call.
    agent = _new_agent()
    call_count = {"n": 0}

    def always_tool_call(**kwargs):
        call_count["n"] += 1
        return _fake_response(tool_calls=[_fake_tool_call(f"call_{call_count['n']}", "nonexistent_tool", "{}")])

    agent.client.chat.completions.create = always_tool_call
    reply = agent.send("loop forever")

    assert reply == "[stopped: too many tool calls in a row without a final answer]"
    assert call_count["n"] == MAX_TOOL_ROUNDS
    assert agent.messages[-1] == {"role": "assistant", "content": reply}


def test_send_executes_tool_and_feeds_result_back():
    agent = _new_agent()
    responses = [
        _fake_response(tool_calls=[_fake_tool_call("call_1", "test_hypothesis", '{"name": "h1", "group_a": [1,2,3]}')]),
        _fake_response(content="done"),
    ]

    def sequenced(**kwargs):
        return responses.pop(0)

    agent.client.chat.completions.create = sequenced
    reply = agent.send("run a test")
    assert reply == "done"
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "ERROR" in tool_messages[0]["content"]  # missing required "group_b" -> handled, not a crash


def test_usage_accumulates_across_multiple_requests():
    agent = _new_agent()
    responses = [
        _fake_response(
            tool_calls=[_fake_tool_call("call_1", "nonexistent_tool", "{}")],
            usage=_fake_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
        _fake_response(content="done", usage=_fake_usage(prompt_tokens=20, completion_tokens=8, total_tokens=28)),
    ]
    agent.client.chat.completions.create = lambda **kwargs: responses.pop(0)
    agent.send("question")

    assert agent.usage["request_count"] == 2
    assert agent.usage["prompt_tokens"] == 30
    assert agent.usage["completion_tokens"] == 13
    assert agent.usage["total_tokens"] == 43


def test_usage_can_be_seeded_from_a_resumed_chat():
    agent = Agent(
        api_key="dummy-test-key",
        base_url="https://example.invalid/v1",
        initial_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "request_count": 3},
    )
    assert agent.usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "request_count": 3}


def test_confirm_callback_can_deny_a_tool_call():
    agent = _new_agent()
    agent.permission_level = "confirm"
    agent.confirm_callback = lambda name, tool_input: False  # deny everything

    responses = [
        _fake_response(tool_calls=[_fake_tool_call("call_1", "get_price_history", '{"symbol": "AAPL"}')]),
        _fake_response(content="acknowledged"),
    ]
    agent.client.chat.completions.create = lambda **kwargs: responses.pop(0)
    agent.send("get AAPL data")

    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "denied" in tool_messages[0]["content"]


def test_confirm_callback_can_approve_a_tool_call():
    agent = _new_agent()
    agent.permission_level = "confirm"
    seen_calls = []
    agent.confirm_callback = lambda name, tool_input: seen_calls.append((name, tool_input)) or True

    responses = [
        _fake_response(tool_calls=[_fake_tool_call("call_1", "test_hypothesis", '{"name": "h1", "group_a": [1]}')]),
        _fake_response(content="done"),
    ]
    agent.client.chat.completions.create = lambda **kwargs: responses.pop(0)
    agent.send("run a test")

    assert seen_calls == [("test_hypothesis", {"name": "h1", "group_a": [1]})]


def test_confirm_mode_without_callback_falls_back_to_auto_approve():
    # Documented degrade-gracefully behavior: "confirm" with no callback
    # wired up should not silently hang forever waiting for an answer that
    # will never come.
    agent = _new_agent()
    agent.permission_level = "confirm"
    assert agent.confirm_callback is None

    responses = [
        _fake_response(tool_calls=[_fake_tool_call("call_1", "nonexistent_tool", "{}")]),
        _fake_response(content="done"),
    ]
    agent.client.chat.completions.create = lambda **kwargs: responses.pop(0)
    reply = agent.send("question")
    assert reply == "done"
