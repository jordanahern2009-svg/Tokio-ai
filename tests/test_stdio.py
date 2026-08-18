"""Regression tests for the Windows codepage crash in the plain REPL.

A model reply containing a single character outside the console's codepage
(an arrow, an en dash, a "greater than or equal" sign -- all routine in LLM
output) raised UnicodeEncodeError from `print()`. In cli.py that print sits
outside the try/except around the API call, so it terminated the whole
session and lost the conversation.
"""

from __future__ import annotations

import io
import sys

from tokio_ai._stdio import force_utf8_stdio

MODEL_FLAVOURED_TEXT = "volume_ratio ≥ 2.0 — gap → fade, “not significant”"


def _cp1252_stream() -> io.TextIOWrapper:
    """A stdout that behaves like a default Windows console."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


def test_cp1252_stdout_cannot_print_model_output_without_the_fix():
    """Pin the actual failure, so the fix below is testing something real."""
    stream = _cp1252_stream()
    try:
        stream.write(MODEL_FLAVOURED_TEXT)
        stream.flush()
    except UnicodeEncodeError:
        return
    raise AssertionError("expected cp1252 to reject this text")


def test_force_utf8_stdio_makes_model_output_printable(monkeypatch):
    stream = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", _cp1252_stream())

    force_utf8_stdio()

    sys.stdout.write(MODEL_FLAVOURED_TEXT)  # must not raise
    sys.stdout.flush()
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"


def test_force_utf8_stdio_survives_a_stream_without_reconfigure(monkeypatch):
    class Bare:
        pass

    monkeypatch.setattr(sys, "stdout", Bare())
    monkeypatch.setattr(sys, "stderr", Bare())
    force_utf8_stdio()  # must not raise


def test_force_utf8_stdio_survives_a_closed_stream(monkeypatch):
    stream = _cp1252_stream()
    stream.close()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", _cp1252_stream())
    force_utf8_stdio()  # must not raise


def test_replaces_rather_than_drops_unrenderable_output(monkeypatch):
    """errors='replace' is the point: degrade a glyph, never lose the answer."""
    stream = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", _cp1252_stream())
    force_utf8_stdio()
    assert sys.stdout.errors == "replace"
