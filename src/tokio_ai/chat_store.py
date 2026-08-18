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
import os
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
    for path in sorted(STORE_DIR.glob("*.json")):
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
    # Write to a sibling temp file and atomically replace, rather than
    # truncating the real file first. A crash (or a Ctrl+C) partway through
    # a direct write leaves a half-written JSON file behind, which is a
    # silently destroyed conversation -- list_chats() already has to skip
    # unparseable files, which is the symptom, not the fix. os.replace is
    # atomic on both POSIX and Windows.
    path = _chat_path(chat_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_chat(chat_id: str) -> tuple[list[dict], TestLedger, dict]:
    data = json.loads(_chat_path(chat_id).read_text(encoding="utf-8"))
    ledger = TestLedger()
    known = set(PermutationResult.__dataclass_fields__)
    for entry in data.get("ledger", []):
        # Drop keys this version doesn't know about instead of raising:
        # a chat saved by a newer TokIO should still open in an older one,
        # and a single unexpected field shouldn't cost the whole history.
        fields = {k: v for k, v in entry["result"].items() if k in known}
        ledger.tests.append(RecordedTest(entry["name"], PermutationResult(**fields)))
    return data.get("messages", []), ledger, data.get("usage", {})


def delete_chat(chat_id: str) -> None:
    path = _chat_path(chat_id)
    if path.exists():
        path.unlink()


def _result_to_dict(result: PermutationResult) -> dict:
    # Derived from the dataclass fields rather than hand-listed, so adding a
    # field to PermutationResult can't silently stop being persisted.
    return {name: getattr(result, name) for name in PermutationResult.__dataclass_fields__}
