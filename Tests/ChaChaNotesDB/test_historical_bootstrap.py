"""Guards for the historical-bootstrap primitive (historical_bootstrap.py).

Successor to the rollback-registry guards (``test_schema_rollback.py``,
retired with the registry in task-16840). The old sweep's oracle compared the
hand-maintained removal registry against the migration chain — two copies of
the same knowledge — and its ratchet forced a registry edit on every schema
bump. Both die with the registry. What remains worth guarding is the REAL
upgrade matrix, and this sweep tests exactly that: for every version the
chain can stop at, bootstrap a genuinely vN-shaped DB (the production chain
itself builds it under a patched ``_CURRENT_SCHEMA_VERSION``), reopen it so
the chain replays to current, and require the result to match a straight
fresh bootstrap on object inventory (type, name) PLUS per-table column sets
(the oracle depth the task-15765 review added: columns are not sqlite_master
rows, so an object-only comparison is blind to column loss; membership is
compared as a SET, not by position).

Honest scope — what this sweep catches and what it cannot:

* CAUGHT: a migration that fails when entered from a genuine historical
  stop point (the exact path a user's old DB takes), version-stamp and
  dispatch wiring defects (a step that stamps past its declared target
  fails the bootstrap's final version check by name), and any divergence
  between the stop-at-vN-then-resume path and the straight-through path.
* IMPOSSIBLE BY CONSTRUCTION: the "table already exists" collision class
  that produced task-15730/15765/16197 — a genuinely vN DB carries no baked
  future artifacts to collide with, so no future migration can break these
  fixtures and nothing needs declaring per version.
* NOT CAUGHT HERE: a defect seeded inside the chain itself (an emptied
  migration step, a wrongly dropped column) that affects the fresh path and
  the replay path identically — both sides of the parity oracle run the same
  chain, so no chain-derived oracle can see it. Those defects are caught
  only where a CONSUMER test pins the artifact (the marks-shape assertions,
  the note-folder suite, the dictionary backfill test — verified born-red in
  task-16840); an artifact no consumer asserts is pinned by NOTHING here —
  the 16840 review seeded an index deletion into the chain and everything
  stayed green (18 of 28 migration-created indexes were unreferenced in
  Tests/). The retired registry was equally blind to this class (its
  removals were DROP IF EXISTS), so nothing was lost. For INDEXES that
  disclosed hole is now covered elsewhere: ``test_index_census.py``
  (task-19045) pins the absolute index-name set, UNIQUE flags, and column
  tuples against a hand-maintained literal, which reds on exactly the
  seeded-deletion shape this sweep must stay green on; for TRIGGERS and
  COLUMNS, ``test_trigger_census.py`` (task-19565) and the
  ``EXPECTED_TABLE_COLUMNS`` literal asserted INSIDE this sweep (also
  task-19565) close the same hole. Other chain-seeded artifact classes
  still rely on consumer tests.
"""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.ChaChaNotesDB.expected_table_columns import EXPECTED_TABLE_COLUMNS
from Tests.ChaChaNotesDB.historical_bootstrap import (
    MINIMUM_BOOTSTRAP_VERSION,
    SCHEMA_NAME,
    chachanotes_db_at_version,
    open_current_chachanotes_from_legacy,
)


@pytest.fixture(scope="module")
def fresh_template_db(tmp_path_factory):
    """Bootstrap one straight current-version DB; sweep cases compare to it."""
    path = tmp_path_factory.mktemp("chacha_template") / "template.sqlite"
    db = CharactersRAGDB(str(path), client_id="template-client")
    db.close_connection()
    return path


def _schema_objects(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Object inventory plus per-table column membership.

    Each table contributes a ``("column", "<table>.<column>")`` entry per
    column, so column loss (invisible in sqlite_master) fails the parity
    assertion by name. Membership is a set on purpose — column ORDER is not
    part of the contract this sweep pins.
    """
    objects = {
        (row[0], row[1])
        for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    }
    for object_type, name in sorted(objects):
        if object_type != "table":
            continue
        objects.update(
            ("column", f"{name}.{column_row[1]}")
            for column_row in conn.execute(f'PRAGMA table_info("{name}")')
        )
    return objects


def _assert_columns_match_declared_expectation(
    conn: sqlite3.Connection, side: str
) -> None:
    """Compare live table->column sets against the hand-declared literal.

    task-19565: the fresh-vs-replay comparison above can never fail for a
    defect seeded inside the chain itself (both sides run the same chain, so
    the comparison is the identity on it). This assertion is the
    independently-declared expectation: a column dropped from BOTH sides by
    a bad migration, or added to both by an unintended one, fails here.
    Applied to the fresh template AND each replayed DB, so the declared
    shape is pinned for both construction paths.
    """
    live: dict[str, frozenset[str]] = {}
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ):
        if name.endswith(("_fts_config", "_fts_data", "_fts_docsize", "_fts_idx")):
            continue  # SQLite-managed FTS shadow internals (see literal module)
        live[name] = frozenset(
            column_row[1] for column_row in conn.execute(f'PRAGMA table_info("{name}")')
        )
    missing_tables = sorted(set(EXPECTED_TABLE_COLUMNS) - set(live))
    assert not missing_tables, (
        f"[{side}] tables declared in EXPECTED_TABLE_COLUMNS are missing from "
        f"the live schema: {missing_tables}. A migration dropped them, or the "
        f"literal in Tests/ChaChaNotesDB/expected_table_columns.py is stale."
    )
    unexpected_tables = sorted(set(live) - set(EXPECTED_TABLE_COLUMNS))
    assert not unexpected_tables, (
        f"[{side}] live tables not declared in EXPECTED_TABLE_COLUMNS: "
        f"{unexpected_tables}. Pin them (expected_table_columns.py) in the "
        f"same commit as the migration that adds them, or drop them."
    )
    divergent = [
        f"{table}: declared-only={sorted(EXPECTED_TABLE_COLUMNS[table] - columns)} "
        f"live-only={sorted(columns - EXPECTED_TABLE_COLUMNS[table])}"
        for table, columns in sorted(live.items())
        if columns != EXPECTED_TABLE_COLUMNS[table]
    ]
    assert not divergent, (
        f"[{side}] live column sets diverge from the independently-declared "
        f"EXPECTED_TABLE_COLUMNS literal (task-19565: the fresh-vs-replay "
        f"comparison cannot see a column dropped from both sides — this "
        f"can):\n" + "\n".join(divergent)
    )


@pytest.mark.parametrize(
    "stop_version",
    range(MINIMUM_BOOTSTRAP_VERSION, CharactersRAGDB._CURRENT_SCHEMA_VERSION),
)
def test_bootstrap_at_version_then_replay_matches_fresh_bootstrap(
    fresh_template_db, tmp_path, stop_version
):
    db_path = tmp_path / f"stop_v{stop_version}.sqlite"
    with chachanotes_db_at_version(db_path, stop_version) as db:
        recorded = (
            db.get_connection()
            .execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (SCHEMA_NAME,),
            )
            .fetchone()[0]
        )
        assert recorded == stop_version

    migrated = open_current_chachanotes_from_legacy(
        db_path, client_id="sweep-client"
    )
    try:
        migrated_conn = migrated.get_connection()
        version = migrated_conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        ).fetchone()[0]
        assert version == migrated._CURRENT_SCHEMA_VERSION

        fresh_conn = sqlite3.connect(fresh_template_db)
        try:
            fresh_objects = _schema_objects(fresh_conn)
            _assert_columns_match_declared_expectation(fresh_conn, "fresh template")
        finally:
            fresh_conn.close()
        replayed_objects = _schema_objects(migrated_conn)
        _assert_columns_match_declared_expectation(migrated_conn, f"replay from v{stop_version}")
        assert replayed_objects == fresh_objects, (
            f"bootstrap at v{stop_version} + replay diverged from a fresh "
            f"bootstrap: missing={sorted(fresh_objects - replayed_objects)} "
            f"extra={sorted(replayed_objects - fresh_objects)}"
        )
    finally:
        migrated.close_connection()
