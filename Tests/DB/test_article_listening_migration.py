"""Real v3/v4 stores and the constructor-free recovery migration."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier, Event

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.recovery_operations import (
    _SUBSCRIPTIONS_SCHEMA,
    _subscriptions_chachanotes_stamps,
    recovery_adapters,
)
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


from Tests.DB.historical_subscriptions import catalog, historical_database


def test_v4_custody_witness_has_strict_sql_and_recovery_bounds(tmp_path):
    from tldw_chatbook.DB.subscriptions_schema import listening_issues

    with closing(SubscriptionsDB(tmp_path / "source.db")) as db:
        c = db.conn
        assert "custody_witness" in {
            r[1] for r in c.execute("PRAGMA table_info(article_audio)")
        }
        c.execute(
            "INSERT INTO article_audio(file_key,profile,status) VALUES (?,?, 'pending_delete')",
            ("0" * 32 + ".wav", "a" * 24),
        )
        for value in (None, "1" * 64):
            c.execute("UPDATE article_audio SET custody_witness=?", (value,))
            assert listening_issues(c) == ()
        for value in ("1" * 64 + "\x00" + "X" * 65536, "A" * 64, b"b" * 64, "0" * 63):
            with pytest.raises(sqlite3.IntegrityError):
                c.execute("UPDATE article_audio SET custody_witness=?", (value,))
            c.execute("PRAGMA ignore_check_constraints=ON")
            c.execute("UPDATE article_audio SET custody_witness=?", (value,))
            queries = []
            c.set_trace_callback(queries.append)
            try:
                assert listening_issues(c) == ("invalid_listening_state",)
            finally:
                c.set_trace_callback(None)
            assert not any(
                "SELECT * FROM article_audio" in statement for statement in queries
            )
            c.execute("PRAGMA ignore_check_constraints=OFF")
        c.rollback()


@pytest.mark.parametrize("existing", [False, True])
def test_concurrent_initializers_install_exact_current_schema(tmp_path, existing):
    path = tmp_path / "subscriptions.db"
    if existing:
        historical_database(path)
    barrier = Barrier(2)

    def construct():
        barrier.wait(timeout=10)
        with closing(SubscriptionsDB(path)) as db:
            assert (
                db.conn.execute("SELECT version FROM schema_version").fetchall()[0][0]
                == 4
            )
            return catalog(db.conn)

    with ThreadPoolExecutor(2) as pool:
        first, second = pool.map(lambda _: construct(), range(2))
    assert first == second
    assert (4, first) in _SUBSCRIPTIONS_SCHEMA
    assert sum(s.startswith("CREATE TABLE listening_") for s in first) == 2


def test_migration_failure_preserves_exact_v3_then_retries(tmp_path):
    path = tmp_path / "historical.db"
    historical_database(path)
    with closing(sqlite3.connect(path)) as con, con:
        before = catalog(con)
        con.execute(
            "CREATE TRIGGER refuse_stamp BEFORE INSERT ON schema_version "
            "WHEN NEW.version = 4 BEGIN SELECT RAISE(ABORT,'test failure'); END"
        )
    with pytest.raises(sqlite3.Error, match="test failure"):
        SubscriptionsDB(path)
    with closing(sqlite3.connect(path)) as con, con:
        con.execute("DROP TRIGGER refuse_stamp")
        assert con.execute("SELECT version FROM schema_version").fetchall() == [(3,)]
        assert catalog(con) == before
    with closing(SubscriptionsDB(path)) as db:
        assert db.conn.execute("SELECT version FROM schema_version").fetchone()[0] == 4


def test_recovery_migrates_only_qualified_lineages_and_preserves_stamps(tmp_path):
    adapter = next(o for o in recovery_adapters() if o.owner_id == "db.subscriptions")
    policy = adapter.schema_policy()
    predecessors = [s for v, s in policy.schema_sql if v == 3]
    successors = [s for v, s in policy.schema_sql if v == 4]
    assert len(successors) == len(predecessors) > 1
    for previous, current in zip(predecessors, successors, strict=True):
        assert set(previous) < set(current)
        assert _subscriptions_chachanotes_stamps(
            previous
        ) == _subscriptions_chachanotes_stamps(current)
    path = tmp_path / "historical.db"
    historical_database(path)
    assert validate_candidate(adapter, path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as con, con:
        assert con.execute("SELECT version FROM schema_version").fetchall() == [(4,)]
        con.execute("CREATE TABLE unexpected(value)")
    assert validate_candidate(adapter, path, Event(), migrate=False)


def test_listening_migration_authorizer_checks_index_target_and_phase():
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

    with closing(sqlite3.connect(":memory:")) as c:
        rules = _Restrictions(c)
        rules.migrating = True
        rules.migration_owner = "db.subscriptions"
        action = sqlite3.SQLITE_CREATE_INDEX
        assert (
            rules.authorize(
                action, "idx_listening_entries_item", "listening_entries", "main", None
            )
            == sqlite3.SQLITE_DENY
        )
        rules.article_listening_migration = True
        assert (
            rules.authorize(
                action, "idx_listening_entries_item", "listening_entries", "main", None
            )
            == sqlite3.SQLITE_OK
        )
        assert (
            rules.authorize(
                action, "idx_listening_entries_item", "subscriptions", "main", None
            )
            == sqlite3.SQLITE_DENY
        )
        assert (
            rules.authorize(
                action,
                "idx_listening_entries_item",
                "listening_entries",
                "main",
                "other",
            )
            == sqlite3.SQLITE_DENY
        )
