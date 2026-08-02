"""What ran, on what, with what -- the minimum needed for someone else (or
future you) to reproduce a result rather than just trust it."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version


def stamp() -> str:
    try:
        tokio_version = version("tokio-ai")
    except PackageNotFoundError:
        from .. import __version__ as tokio_version  # editable/dev install fallback

    try:
        openai_version = version("openai")
    except PackageNotFoundError:
        openai_version = "unknown"

    return f"tokio-ai {tokio_version} | Python {platform.python_version()} | openai {openai_version}"
