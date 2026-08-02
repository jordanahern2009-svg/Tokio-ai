import shutil
import tempfile
from pathlib import Path

from tokio_ai import chat_store
from tokio_ai.rigor.ledger import TestLedger
from tokio_ai.rigor.stats import PermutationResult


def _isolated_store():
    """Redirect STORE_DIR to a fresh temp directory for the duration of a test."""
    tmp = Path(tempfile.mkdtemp())
    original = chat_store.STORE_DIR
    chat_store.STORE_DIR = tmp
    return tmp, original


def _restore(original):
    chat_store.STORE_DIR = original


def test_derive_title_uses_first_user_message():
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "What is AAPL's price?"},
    ]
    assert chat_store.derive_title(messages) == "What is AAPL's price?"


def test_derive_title_truncates_long_messages():
    long_text = "x" * 100
    messages = [{"role": "user", "content": long_text}]
    title = chat_store.derive_title(messages)
    assert len(title) == chat_store.TITLE_MAX_LEN + 3  # + "..."
    assert title.endswith("...")


def test_derive_title_falls_back_when_no_user_message():
    assert chat_store.derive_title([{"role": "system", "content": "x"}]) == "New chat"
    assert chat_store.derive_title([]) == "New chat"


def test_save_and_load_round_trips_messages_and_ledger():
    tmp, original = _isolated_store()
    try:
        chat_id = chat_store.new_chat_id()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        ledger = TestLedger()
        ledger.record("h1", PermutationResult(observed_gap=0.05, p_value=0.02, n_a=40, n_b=40, iters=5000, seed=1))

        chat_store.save_chat(chat_id, messages, ledger, usage={"total_tokens": 123})
        loaded_messages, loaded_ledger, loaded_usage = chat_store.load_chat(chat_id)

        assert loaded_messages == messages
        assert len(loaded_ledger.tests) == 1
        assert loaded_ledger.tests[0].name == "h1"
        assert loaded_ledger.tests[0].result.p_value == 0.02
        assert loaded_ledger.tests[0].result.iters == 5000
        assert loaded_usage == {"total_tokens": 123}
    finally:
        _restore(original)
        shutil.rmtree(tmp, ignore_errors=True)


def test_list_chats_sorted_by_most_recently_updated():
    tmp, original = _isolated_store()
    try:
        id_a = chat_store.new_chat_id()
        chat_store.save_chat(id_a, [{"role": "user", "content": "first chat"}], TestLedger())
        id_b = chat_store.new_chat_id()
        chat_store.save_chat(id_b, [{"role": "user", "content": "second chat"}], TestLedger())
        # force distinguishable timestamps without relying on real sleep
        import json
        path_a = chat_store._chat_path(id_a)
        data = json.loads(path_a.read_text(encoding="utf-8"))
        data["updated_at"] = "2020-01-01T00:00:00+00:00"
        path_a.write_text(json.dumps(data), encoding="utf-8")

        summaries = chat_store.list_chats()
        assert [s.id for s in summaries] == [id_b, id_a]
        assert summaries[0].title == "second chat"
    finally:
        _restore(original)
        shutil.rmtree(tmp, ignore_errors=True)


def test_list_chats_skips_corrupt_files_without_crashing():
    tmp, original = _isolated_store()
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "bad.json").write_text("not valid json{{{", encoding="utf-8")
        good_id = chat_store.new_chat_id()
        chat_store.save_chat(good_id, [{"role": "user", "content": "ok"}], TestLedger())

        summaries = chat_store.list_chats()
        assert len(summaries) == 1
        assert summaries[0].id == good_id
    finally:
        _restore(original)
        shutil.rmtree(tmp, ignore_errors=True)


def test_list_chats_empty_when_store_dir_missing():
    tmp, original = _isolated_store()
    shutil.rmtree(tmp, ignore_errors=True)  # directory doesn't exist at all
    try:
        assert chat_store.list_chats() == []
    finally:
        _restore(original)


def test_delete_chat_removes_file():
    tmp, original = _isolated_store()
    try:
        chat_id = chat_store.new_chat_id()
        chat_store.save_chat(chat_id, [{"role": "user", "content": "x"}], TestLedger())
        assert chat_store._chat_path(chat_id).exists()
        chat_store.delete_chat(chat_id)
        assert not chat_store._chat_path(chat_id).exists()
    finally:
        _restore(original)
        shutil.rmtree(tmp, ignore_errors=True)


def test_delete_chat_nonexistent_does_not_raise():
    tmp, original = _isolated_store()
    try:
        chat_store.delete_chat("does-not-exist")  # should not raise
    finally:
        _restore(original)
        shutil.rmtree(tmp, ignore_errors=True)
