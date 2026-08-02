"""Tiny .env loader, stdlib only -- avoids adding python-dotenv as a dependency."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path | None = None) -> None:
    path = path or Path.cwd() / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        # Strip one layer of matching quotes -- "KEY=\"value\"" is a common
        # .env convention (copied from shell export syntax, other dotenv
        # examples) that would otherwise leave literal quote characters in
        # the value and silently break auth.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key.strip(), val)
