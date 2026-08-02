"""Local, on-disk chat history: multiple named, resumable conversations.

Stored as one JSON file per chat under ~/.tokio_ai/chats/ -- no database
dependency, just files, consistent with the project's minimal-dependency
approach everywhere else. Each file holds the raw message list (already
JSON-serializable dicts, same shape the OpenAI-compatible API expects back)
plus the TestLedger's recorded tests, so resuming a chat also resumes its
multiple-testing correction state -- picking a conversation back up should
behave as if you never left, including for that.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .rigor.ledger import RecordedTest, TestLedger
from .rigor.stats import PermutationResult

STORE_DIR = Path.home() / ".tokio_ai" / "chats"
TITLE_MAX_LEN = 50


@dataclass(frozen=True)
class ChatSummary:
    id: str
    title: str
    updated_at: str
    message_count: int


def _chat_path(chat_id: str) -> Path:
    return STORE_DIR / f"{chat_id}.json"


def new_chat_id() -> str:
    return uuid.uuid4().hex[:12]


def derive_title(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            text = m["content"].strip()
            return text[:TITLE_MAX_LEN] + ("..." if len(text) > TITLE_MAX_LEN else "")
    return "New chat"


def list_chats() -> list[ChatSummary]:
    if not STORE_DIR.exists():
        return []
    summaries = []
    for path in STORE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # a corrupt/partial file shouldn't take down the whole chat list
        user_turns = sum(1 for m in data.get("messages", []) if m.get("role") == "user")
        summaries.append(
            ChatSummary(
                id=data.get("id", path.stem),
                title=data.get("title") or "New chat",
                updated_at=data.get("updated_at", ""),
                message_count=user_turns,
            )
        )
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


def save_chat(
    chat_id: str,
    messages: list[dict],
    ledger: TestLedger,
    usage: dict | None = None,
) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "id": chat_id,
        "title": derive_title(messages),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "ledger": [
            {"name": t.name, "result": _result_to_dict(t.result)} for t in ledger.tests
        ],
        "usage": usage or {},
    }
    _chat_path(chat_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_chat(chat_id: str) -> tuple[list[dict], TestLedger, dict]:
    data = json.loads(_chat_path(chat_id).read_text(encoding="utf-8"))
    ledger = TestLedger()
    for entry in data.get("ledger", []):
        ledger.tests.append(RecordedTest(entry["name"], PermutationResult(**entry["result"])))
    return data.get("messages", []), ledger, data.get("usage", {})


def delete_chat(chat_id: str) -> None:
    path = _chat_path(chat_id)
    if path.exists():
        path.unlink()


def _result_to_dict(result: PermutationResult) -> dict:
    return {
        "observed_gap": result.observed_gap,
        "p_value": result.p_value,
        "n_a": result.n_a,
        "n_b": result.n_b,
        "iters": result.iters,
        "seed": result.seed,
    }
