"""TokIO AI: an open-source financial research agent that runs every claim
through honest statistical testing before trusting it."""

__version__ = "0.4.0"

from .check import CheckResult, check  # noqa: E402  (needs __version__ defined first)
from .rigor.ledger import TestLedger  # noqa: E402

__all__ = ["check", "CheckResult", "TestLedger", "__version__"]
