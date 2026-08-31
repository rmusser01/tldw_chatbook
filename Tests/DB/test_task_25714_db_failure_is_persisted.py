"""TASK-25714: a database failure must reach the log users are told to check.

Console refuses a send with "check the app log for the database error if it
keeps happening". The persistent log admits ONLY records marked metadata-only
by `persist_event` (PersistentDiagnosticFilter) -- every ordinary
`logger.error` is structurally excluded. So that instruction pointed at a file
that, by design, could not contain the fault. Verified live: a corrupt index
made Console unusable and the app log for that session held 15 INFO lines and
no trace of it.

The filter is a deliberate privacy boundary (exception text can carry secrets)
and must not be weakened. The fix is to emit a metadata-only event instead.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def persisted_records(monkeypatch):
    seen: list[tuple[str, dict]] = []

    def _record(component, event, *, level=logging.INFO, **fields):
        seen.append((event, fields))

    import tldw_chatbook.DB.ChaChaNotes_DB as chacha

    monkeypatch.setattr(chacha, "persist_event", _record, raising=False)
    return seen


def test_schema_failure_is_persisted_with_metadata_only(persisted_records, tmp_path):
    """A failed schema init must name itself in the persistent log."""
    import sqlite3

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db_path = tmp_path / "corrupt.db"
    # A database whose schema version is far beyond what this code supports is
    # the cheapest reproducible "cannot be opened" fault.
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE db_schema_version (schema_name TEXT PRIMARY KEY, version INTEGER)"
    )
    conn.execute(
        "INSERT INTO db_schema_version VALUES ('rag_char_chat_schema', 999999)"
    )
    conn.commit()
    conn.close()

    with pytest.raises(Exception):
        CharactersRAGDB(db_path=str(db_path), client_id="t25714")

    events = [name for name, _ in persisted_records]
    assert "database_open_failed" in events, (
        "a database that cannot be opened must emit a persistent, metadata-only "
        f"event; persisted events were {events}"
    )
    fields = dict(
        persisted_records[
            [n for n, _ in persisted_records].index("database_open_failed")
        ][1]
    )
    assert "error_type" in fields
    assert "repairable" in fields
    # Metadata only: no free-text message that could carry a path or secret.
    assert all(
        "/" not in str(value) and "\\\\" not in str(value) for value in fields.values()
    ), f"persisted fields must stay metadata-only, got {fields}"
