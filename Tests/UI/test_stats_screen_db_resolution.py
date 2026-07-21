"""Stats screen DB resolution regression tests.

The Stats screen used to probe a stale ``characters_rag_db`` attribute that
no longer exists on ``TldwCli``, so every visit reported "Database connection
not initialized" and the destination header sat on Error. It must use the
canonical access path instead: prefer ``app_instance.chachanotes_db``, fall
back to ``notes_service.db`` (mirroring the Library screen's resolvers).
"""

from types import SimpleNamespace

from tldw_chatbook.UI.Screens.stats_screen import StatsScreen


def test_resolve_stats_db_prefers_chachanotes_db():
    db = object()
    app = SimpleNamespace(chachanotes_db=db, notes_service=None)
    assert StatsScreen._resolve_stats_db(app) is db


def test_resolve_stats_db_falls_back_to_notes_service_db():
    db = object()
    app = SimpleNamespace(chachanotes_db=None, notes_service=SimpleNamespace(db=db))
    assert StatsScreen._resolve_stats_db(app) is db


def test_resolve_stats_db_returns_none_when_unavailable():
    app = SimpleNamespace()
    assert StatsScreen._resolve_stats_db(app) is None
