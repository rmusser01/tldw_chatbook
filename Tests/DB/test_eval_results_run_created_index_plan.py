"""task-34399 item 4: ``idx_eval_results_run_created``.

``get_run_results`` pages ``WHERE run_id = ? ORDER BY created_at ASC``;
with only the bare ``(run_id)`` index every page re-sorted the run's rows.
The composite index ships through this store's house migration pattern:
``SCHEMA_VERSION`` bumps 6 -> 7 (dev's own v6 was the TASK-19566 F8
version-column drop), the fresh-schema index block gains the index, and
``_migrate_schema`` recreates it for existing v6 databases.
"""

import pytest

from tldw_chatbook.DB.Evals_DB import SCHEMA_VERSION, EvalsDB


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "evals-run-created.db")


def test_index_exists_on_a_fresh_database(db_path):
    db = EvalsDB(db_path=db_path, client_id="test")

    columns = [
        row[2]
        for row in db.get_connection().execute(
            "PRAGMA index_info(idx_eval_results_run_created)"
        )
    ]

    assert columns == ["run_id", "created_at"]


def test_opening_a_v6_database_migrates_and_recreates_the_index(db_path):
    """The exact upgrade an existing user's database takes: v6 file loses
    nothing, the migration step adds the index, user_version lands on the
    live SCHEMA_VERSION."""
    first = EvalsDB(db_path=db_path, client_id="test")
    conn = first.get_connection()
    with conn:
        conn.execute("DROP INDEX idx_eval_results_run_created")
        conn.execute("PRAGMA user_version = 6")
    # Sanity: the fixture really removed it before reopening.
    assert (
        conn.execute("PRAGMA index_info(idx_eval_results_run_created)").fetchone()
        is None
    )

    second = EvalsDB(db_path=db_path, client_id="test")
    migrated = second.get_connection()

    assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    columns = [
        row[2]
        for row in migrated.execute(
            "PRAGMA index_info(idx_eval_results_run_created)"
        )
    ]
    assert columns == ["run_id", "created_at"]


def test_run_results_page_uses_the_index_without_stats(db_path):
    """The plan pin (scripts/check_index_plan_pins.py).

    The plan is only evidence when captured the way production runs it:
    Evals_DB never runs ANALYZE, so no user's database has a sqlite_stat1
    and the planner works from default row-count estimates. ``TEMP B-TREE``
    reappearing is the re-sort this index exists to remove.
    """
    db = EvalsDB(db_path=db_path, client_id="test")
    conn = db.get_connection()

    assert (
        conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='sqlite_stat1'"
        ).fetchone()
        is None
    ), (
        "this fixture must reproduce the no-stats production state; a plan "
        "captured with sqlite_stat1 present is not the plan users run"
    )

    page_sql = (
        "SELECT * FROM eval_results WHERE run_id = ?"
        " ORDER BY created_at ASC LIMIT ? OFFSET ?"
    )
    plan = " | ".join(
        str(row[3])
        for row in conn.execute("EXPLAIN QUERY PLAN " + page_sql, ("run-1", 1000, 0))
    )

    assert "idx_eval_results_run_created" in plan
    assert "SCAN eval_results" not in plan, plan
    assert "TEMP B-TREE" not in plan, plan
