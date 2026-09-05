from agentic_sdlc.memory.store import MemoryStore


def test_upsert_and_get(tmp_path):
    store = MemoryStore(db_path=tmp_path / "mem.sqlite3")
    store.upsert("repo/a", "issue:app.py:missing_docstring", {"resolved": False, "note": "flagged twice"})
    record = store.get("repo/a", "issue:app.py:missing_docstring")
    assert record["resolved"] is False
    assert record["note"] == "flagged twice"
    store.close()


def test_missing_key_returns_none(tmp_path):
    store = MemoryStore(db_path=tmp_path / "mem.sqlite3")
    assert store.get("repo/a", "nonexistent") is None
    store.close()


def test_all_for_repo_scopes_correctly(tmp_path):
    store = MemoryStore(db_path=tmp_path / "mem.sqlite3")
    store.upsert("repo/a", "k1", {"x": 1})
    store.upsert("repo/b", "k1", {"x": 2})
    records = store.all_for_repo("repo/a")
    assert len(records) == 1
    assert records[0].value == {"x": 1}
    store.close()
