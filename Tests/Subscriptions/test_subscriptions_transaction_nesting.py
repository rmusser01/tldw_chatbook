"""`SubscriptionsDB.transaction()` must be safe to nest (TASK-19562 part C).

It used to commit unconditionally on exit, so an INNER `with
self.transaction()` durably committed the OUTER transaction's work too: a
later failure in the outer scope could no longer roll back what the inner
block had already written. Silent partial persistence, no error anywhere.

Recorded for accuracy, replacing an earlier claim in this docstring that
`record_check_result` did NOT nest ("observed depth 1"). Re-instrumented per
argument shape, it does:

    record_check_result WITH stats    -> 2 entries, depths [1, 2]
    record_check_result WITHOUT stats -> 1 entry,  depths [1]

The route is `record_check_result` -> `_update_subscription_stats` ->
`update_subscription_stats`, which opens its own transaction for the
`subscription_stats` upsert, and `execute_run` always passes stats. The
earlier measurement can only have taken the `stats=None` path. So the hazard
was **live**, not latent -- the daily-statistics write was committing the
enclosing subscription-health UPDATE -- and
`test_record_check_result_with_stats_is_one_atomic_unit` below pins the real
call site, not just a synthetic one.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _names(database: SubscriptionsDB) -> set[str]:
    with database.transaction() as conn:
        return {
            row["name"]
            for row in conn.execute("SELECT name FROM subscriptions").fetchall()
        }


def test_failure_after_a_nested_block_rolls_the_whole_unit_back(db):
    """The defect, stated as behaviour: the inner write must not survive."""
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
                ("outer", "rss", "https://e.invalid/outer.xml"),
            )
            with db.transaction() as inner:
                inner.execute(
                    "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
                    ("inner", "rss", "https://e.invalid/inner.xml"),
                )
            # Fails AFTER the inner block exited. Pre-fix, the inner exit had
            # already committed both rows, so this rollback did nothing.
            raise RuntimeError("outer scope fails")

    surviving = _names(db)
    assert "inner" not in surviving, (
        "the nested block's write survived a failure in the outer scope: its "
        "exit committed the outer transaction early"
    )
    assert "outer" not in surviving


def test_nested_success_still_commits_once_the_outer_block_exits(db):
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
            ("outer", "rss", "https://e.invalid/outer.xml"),
        )
        with db.transaction() as inner:
            inner.execute(
                "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
                ("inner", "rss", "https://e.invalid/inner.xml"),
            )

    assert {"outer", "inner"} <= _names(db)


def test_depth_returns_to_zero_after_a_raise(db):
    """A stranded depth would make every later transaction a no-op joiner."""
    with pytest.raises(RuntimeError):
        with db.transaction():
            raise RuntimeError("boom")

    assert getattr(db._local, "transaction_depth", 0) == 0

    # And the connection is still usable as an outermost transaction.
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
            ("after", "rss", "https://e.invalid/after.xml"),
        )
    assert "after" in _names(db)


def test_unnested_transactions_are_unchanged(db):
    """AC: no behaviour change for the ordinary single-level case."""
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
            ("solo", "rss", "https://e.invalid/solo.xml"),
        )
    assert "solo" in _names(db)

    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
                ("doomed", "rss", "https://e.invalid/doomed.xml"),
            )
            raise RuntimeError("boom")
    assert "doomed" not in _names(db)


def test_record_check_result_with_stats_really_nests(db, monkeypatch):
    """The live call site, measured -- not assumed either way.

    The task rated this LATENT on the strength of an instrumentation that
    saw depth 1. That measurement took the `stats=None` path;
    `execute_run` never does. Pinning the observed depths here means the
    next person does not have to re-derive which branch nests.
    """
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    source_id = db.add_subscription(
        name="s", type="rss", source="https://e.invalid/f.xml"
    )
    depths: list[int] = []
    original = SubscriptionsDB.transaction

    import contextlib

    @contextlib.contextmanager
    def traced(self):
        depths.append(getattr(self._local, "transaction_depth", 0) + 1)
        with original(self) as conn:
            yield conn

    monkeypatch.setattr(SubscriptionsDB, "transaction", traced)

    db.record_check_result(source_id, stats={"response_time_ms": 5})
    assert depths == [1, 2], (
        f"expected record_check_result to nest via the statistics upsert, "
        f"observed depths {depths}"
    )

    depths.clear()
    db.record_check_result(source_id, stats=None)
    assert depths == [1], (
        "the stats=None path does NOT nest -- this is the branch the "
        f"original 'does not nest' measurement must have taken: {depths}"
    )


def test_record_check_result_with_stats_is_one_atomic_unit(db):
    """A failure after the nested statistics write must undo it.

    Red against the unnested context manager: the inner
    `update_subscription_stats` commit persists the `subscription_stats` row
    (and the health UPDATE with it), so the rollback below finds nothing to
    undo.
    """
    source_id = db.add_subscription(
        name="s", type="rss", source="https://e.invalid/f.xml"
    )

    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            db.record_check_result(source_id, stats={"response_time_ms": 5})
            conn.execute(
                "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
                ("sibling", "rss", "https://e.invalid/sibling.xml"),
            )
            raise RuntimeError("the enclosing unit of work fails")

    rows = db.conn.execute(
        "SELECT COUNT(*) FROM subscription_stats WHERE subscription_id = ?",
        (source_id,),
    ).fetchone()
    assert rows[0] == 0, (
        "the nested statistics write survived a failure in the enclosing "
        "transaction: record_check_result's inner commit ended the outer one"
    )
    assert "sibling" not in _names(db)
