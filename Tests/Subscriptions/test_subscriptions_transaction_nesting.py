"""`SubscriptionsDB.transaction()` must be safe to nest (TASK-19562 part C).

It used to commit unconditionally on exit, so an INNER `with
self.transaction()` durably committed the OUTER transaction's work too: a
later failure in the outer scope could no longer roll back what the inner
block had already written. Silent partial persistence, no error anywhere.

Recorded for accuracy: the call site the task named (`record_check_result`)
was instrumented and found NOT to nest -- observed depth 1. The hazard was
latent, not live. These tests make it structurally impossible instead of
relying on nobody ever nesting.
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
