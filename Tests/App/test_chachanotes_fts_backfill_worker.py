"""The boot FTS backfill never creates the ChaChaNotes database (TASK-38462).

A missing database has no messages to index, and creating one (schema v0 to
current, plus built-in seeding) is a multi-second step that no cancel can
stop. On Windows that creation outlived the app's exit and the 3 s worker join
in test teardown once timers stopped slowing the test bodies.

The backfill's existence check and ``get_chachanotes_db_lazy`` both call
``config.get_chachanotes_db_path()`` with no arguments, so they name the same
file; these tests stub that one function to pin the branch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook import app_feature_glue, config
from tldw_chatbook.app import TldwCli


@pytest.fixture
def lazy_calls(tmp_path, monkeypatch) -> list[str]:
    """Point ChaChaNotes at a scratch path and record every lazy-open attempt."""
    calls: list[str] = []
    monkeypatch.setattr(config, "chachanotes_db", None)
    monkeypatch.setattr(
        config,
        "get_chachanotes_db_path",
        lambda **_kwargs: tmp_path / "ChaChaNotes.db",
    )
    monkeypatch.setattr(
        app_feature_glue, "get_chachanotes_db_lazy", lambda: calls.append("open")
    )
    return calls


def test_backfill_skips_a_profile_with_no_chachanotes_database(lazy_calls):
    TldwCli._backfill_chachanotes_messages_fts(SimpleNamespace())
    assert lazy_calls == []


def test_backfill_opens_an_existing_database_file(tmp_path, lazy_calls):
    (tmp_path / "ChaChaNotes.db").touch()
    TldwCli._backfill_chachanotes_messages_fts(SimpleNamespace())
    assert lazy_calls == ["open"]


def test_backfill_uses_an_already_open_database(monkeypatch, lazy_calls):
    monkeypatch.setattr(config, "chachanotes_db", object())
    TldwCli._backfill_chachanotes_messages_fts(SimpleNamespace())
    assert lazy_calls == ["open"]
