"""Make stdout/stderr safe for model output on Windows.

Python picks the console's codepage for stdout, which on a default Windows
install is cp1252 -- a 256-character set. Language models emit characters
outside it constantly: the "greater than or equal" sign when describing a
threshold, en/em dashes, curly quotes, arrows. Printing one of those raises
UnicodeEncodeError, and in the plain REPL that happened *outside* the
try/except around the API call, so a single arrow character killed the
session and lost the conversation.

This is the third time the same codepage assumption has bitten this project
(the bundled S&P 500 CSV, on both the write and the read side). The lesson
each time: never let the platform's default encoding be the one that
decides, and prefer degrading a character over dropping the output.
"""

from __future__ import annotations

import sys


def force_utf8_stdio() -> None:
    """Re-encode stdout/stderr as UTF-8, replacing anything unrenderable.

    `errors="replace"` is deliberate belt-and-braces: on a console that
    genuinely cannot render a glyph, a "?" in the output is a far better
    outcome than a traceback over a punctuation mark.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # a redirected/wrapped stream may not have it
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Detached or already-closed stream: nothing to do, and this is
            # never worth failing startup over.
            pass
