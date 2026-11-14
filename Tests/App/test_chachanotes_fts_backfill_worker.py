"""The boot FTS backfill never creates the ChaChaNotes database (TASK-38462).

A missing database has no messages to index, and creating one (schema v0 to
current, plus built-in seeding) is a multi-second step that no cancel can
stop. On Windows that creation outlived the app's exit and the 3 s worker join
in test teardown once timers stopped slowing the test bodies.

Since TASK-36947.6 the backfill only uses the database the app already opened,
``services.storage.chachanotes``; it never opens one itself.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tldw_chatbook import config
from tldw_chatbook.app import TldwCli
from tldw_chatbook.app_services import ProfileStorage


@pytest.fixture
def backfilled(tmp_path, monkeypatch) -> list[object]:
    """Point ChaChaNotes at a scratch path and record every backfilled database."""
    import tldw_chatbook.DB.base_db as base_db
    import tldw_chatbook.DB.chachanotes_fts_backfill as fts_backfill

    calls: list[object] = []
    monkeypatch.setattr(
        config,
        "get_chachanotes_db_path",
        lambda **_kwargs: tmp_path / "ChaChaNotes.db",
    )
    monkeypatch.setattr(base_db, "operation_owned_connection", lambda _db: nullcontext())
    monkeypatch.setattr(
        fts_backfill,
        "backfill_chachanotes_messages_fts",
        lambda db, **_kwargs: calls.append(db),
    )
    return calls


def _app(chachanotes: object | None) -> SimpleNamespace:
    return SimpleNamespace(
        services=SimpleNamespace(storage=ProfileStorage(chachanotes=chachanotes))
    )


def test_backfill_skips_an_app_with_no_chachanotes_database(tmp_path, backfilled):
    (tmp_path / "ChaChaNotes.db").touch()  # a file alone is not an open database
    TldwCli._backfill_chachanotes_messages_fts(_app(None))
    assert backfilled == []


def test_backfill_never_creates_the_database_file(tmp_path, backfilled):
    TldwCli._backfill_chachanotes_messages_fts(_app(None))
    assert not (tmp_path / "ChaChaNotes.db").exists()


def test_backfill_uses_the_database_the_app_opened(backfilled):
    database = object()
    TldwCli._backfill_chachanotes_messages_fts(_app(database))
    assert backfilled == [database]
