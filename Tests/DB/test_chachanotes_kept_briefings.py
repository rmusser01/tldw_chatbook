# test_chachanotes_kept_briefings.py
#
# Task-1780 (kept-briefings): schema + CRUD for kept_briefings/kept_scripts in
# ChaChaNotes. See Docs/superpowers/specs/2026-08-01-kept-briefings-design.md
# and the v28->v29 migration
# (tldw_chatbook/DB/migrations/chachanotes_v28_to_v29_kept_briefings.sql) for
# the schema rationale.
#
# Every test constructs a real CharactersRAGDB rooted at pytest's `tmp_path`
# -- never the live user config/data directory -- per this stream's
# live-DB-safety rule. No probes or ad-hoc script execution; pytest only.
#
# Imports
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import (
    open_current_chachanotes_from_legacy,
)

from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

SCHEMA_NAME = "rag_char_chat_schema"
MIGRATION_SQL_PATH = (
    Path(__file__).parents[2]
    / "tldw_chatbook"
    / "DB"
    / "migrations"
    / "chachanotes_v28_to_v29_kept_briefings.sql"
)


# --- Helpers -----------------------------------------------------------


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _table_names(connection: sqlite3.Connection) -> set:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def _seed_v28_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a real v28 database that never touches the kept_* tables."""

    with monkeypatch.context() as v28_patch:
        v28_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 28)
        db = CharactersRAGDB(path, client_id="migration-seed")
        db.close_connection()


def _extract_table_body(sql_lower: str, table: str) -> str:
    marker = f"create table if not exists {table}("
    start = sql_lower.index(marker) + len(marker)
    end = sql_lower.index(");", start)
    return sql_lower[start:end]


def _make_db(tmp_path: Path, name: str = "kept.sqlite") -> CharactersRAGDB:
    return CharactersRAGDB(tmp_path / name, client_id="kept-briefings-test")


# --- create_kept_briefing / get_kept_briefing / get_kept_briefing_by_source


def test_create_and_get_kept_briefing_round_trips_all_fields(tmp_path: Path) -> None:
    """`covers_from_ts`/`original_created_at`/`kept_at` are declared
    `DATETIME`; every `CharactersRAGDB` connection opens with
    `PARSE_DECLTYPES` plus the process-wide `DATETIME` converter
    (`DB/sqlite_datetime_fix.py`), so a caller-supplied ISO-8601 string with
    an explicit offset/`Z` round-trips as a tz-aware `datetime.datetime` --
    exactly like every other `DATETIME` column in this database (e.g.
    `conversations.created_at`), not as the original string."""
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=101,
            watchlist_name="Tech Watch",
            body_markdown="# Digest\n\nBody text.",
            covers_through_item_id=555,
            covers_from_ts="2026-07-25T00:00:00Z",
            selection_mode="auto",
            model_used="gpt-test",
            item_count=12,
            featured_count=3,
            overflow_count=2,
            origin="manual",
            original_created_at="2026-07-30T10:00:00Z",
        )
        assert isinstance(kept_id, int)

        row = db.get_kept_briefing(kept_id)
        assert row is not None
        assert row["source_briefing_id"] == 101
        assert row["watchlist_name"] == "Tech Watch"
        assert row["body_markdown"] == "# Digest\n\nBody text."
        assert row["covers_through_item_id"] == 555
        assert row["covers_from_ts"] == datetime(2026, 7, 25, 0, 0, tzinfo=timezone.utc)
        assert row["selection_mode"] == "auto"
        assert row["model_used"] == "gpt-test"
        assert row["item_count"] == 12
        assert row["featured_count"] == 3
        assert row["overflow_count"] == 2
        assert row["origin"] == "manual"
        assert row["original_created_at"] == datetime(
            2026, 7, 30, 10, 0, tzinfo=timezone.utc
        )
        assert row["kept_at"]

        assert db.get_kept_briefing_by_source(101) == row
    finally:
        db.close_connection()


def test_create_kept_briefing_accepts_explicit_kept_at(tmp_path: Path) -> None:
    """`kept_at` defaults to `CURRENT_TIMESTAMP`, but the chatbook importer
    (task-1870) needs to preserve the *originating* device's keep time on a
    cross-device import rather than silently re-stamp it with the import
    moment -- an explicit `kept_at` must be honored verbatim, exactly like
    `original_created_at` already is (task-1870 fix-wave F4)."""
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=201,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
            kept_at="2020-01-01T00:00:00Z",
        )
        row = db.get_kept_briefing(kept_id)
        assert row["kept_at"] == datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
    finally:
        db.close_connection()


def test_create_kept_briefing_without_kept_at_uses_column_default(
    tmp_path: Path,
) -> None:
    """Omitting `kept_at` (every caller except the chatbook importer) must
    still fall through to the column's own `DEFAULT CURRENT_TIMESTAMP`."""
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=202,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        row = db.get_kept_briefing(kept_id)
        assert row["kept_at"] is not None
    finally:
        db.close_connection()


def test_create_kept_briefing_applies_defaults_for_optional_fields(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=1,
            watchlist_name=None,
            body_markdown="Body",
            origin="scheduled",
        )
        row = db.get_kept_briefing(kept_id)
        assert row["watchlist_name"] is None
        assert row["covers_through_item_id"] is None
        assert row["covers_from_ts"] is None
        assert row["selection_mode"] is None
        assert row["model_used"] is None
        assert row["item_count"] == 0
        assert row["featured_count"] == 0
        assert row["overflow_count"] == 0
        assert row["original_created_at"] is None
        assert row["origin"] == "scheduled"
    finally:
        db.close_connection()


def test_get_kept_briefing_and_by_source_return_none_when_absent(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    try:
        assert db.get_kept_briefing(9999) is None
        assert db.get_kept_briefing_by_source(9999) is None
    finally:
        db.close_connection()


def test_duplicate_source_briefing_id_raises_conflict_and_leaves_one_row(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    try:
        db.create_kept_briefing(
            source_briefing_id=42,
            watchlist_name="W",
            body_markdown="Body",
            origin="manual",
        )
        with pytest.raises(ConflictError):
            db.create_kept_briefing(
                source_briefing_id=42,
                watchlist_name="W2",
                body_markdown="Body2",
                origin="scheduled",
            )
        assert len(db.list_kept_briefings()) == 1
    finally:
        db.close_connection()


def test_create_kept_briefing_rejects_invalid_origin_before_any_row(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    try:
        with pytest.raises(InputError):
            db.create_kept_briefing(
                source_briefing_id=1,
                watchlist_name="W",
                body_markdown="Body",
                origin="auto",
            )
        assert db.list_kept_briefings() == []
    finally:
        db.close_connection()


def test_origin_check_constraint_enforced_at_schema_level(tmp_path: Path) -> None:
    """The CHECK constraint itself must reject a bad value, independent of
    the Python-side guard in `create_kept_briefing` (bypassed here via a raw
    INSERT)."""
    db = _make_db(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.get_connection().execute(
                "INSERT INTO kept_briefings(source_briefing_id, body_markdown, origin) "
                "VALUES (?, ?, ?)",
                (1, "Body", "auto"),
            )
    finally:
        db.close_connection()


def test_origin_not_null_enforced_at_schema_level(tmp_path: Path) -> None:
    """A CHECK constraint alone permits NULL (SQL NULL comparisons are
    UNKNOWN, not FALSE); `origin` also carries NOT NULL so a NULL origin
    is rejected too."""
    db = _make_db(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.get_connection().execute(
                "INSERT INTO kept_briefings(source_briefing_id, body_markdown, origin) "
                "VALUES (?, ?, NULL)",
                (1, "Body"),
            )
    finally:
        db.close_connection()


# --- list_kept_briefings: ordering + pagination -------------------------


def test_list_kept_briefings_orders_by_kept_at_desc_then_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        ids = [
            db.create_kept_briefing(
                source_briefing_id=i,
                watchlist_name="W",
                body_markdown="B",
                origin="manual",
            )
            for i in (1, 2, 3)
        ]
        # Explicit, distinguishable kept_at values -- id[1] is newest even
        # though it was inserted second, so ordering can only be explained
        # by sorting on kept_at, not on insertion/id order alone.
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE kept_briefings SET kept_at = ? WHERE id = ?",
                ("2026-01-01T00:00:00Z", ids[0]),
            )
            cursor.execute(
                "UPDATE kept_briefings SET kept_at = ? WHERE id = ?",
                ("2026-01-03T00:00:00Z", ids[1]),
            )
            cursor.execute(
                "UPDATE kept_briefings SET kept_at = ? WHERE id = ?",
                ("2026-01-02T00:00:00Z", ids[2]),
            )
        result_ids = [row["id"] for row in db.list_kept_briefings()]
        assert result_ids == [ids[1], ids[2], ids[0]]
    finally:
        db.close_connection()


def test_list_kept_briefings_ties_on_kept_at_break_by_id_descending(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    try:
        ids = [
            db.create_kept_briefing(
                source_briefing_id=i,
                watchlist_name="W",
                body_markdown="B",
                origin="manual",
            )
            for i in (1, 2, 3)
        ]
        # Force an identical kept_at across all rows: the only thing left
        # to order by is the `id DESC` tiebreak.
        with db.transaction() as cursor:
            cursor.execute("UPDATE kept_briefings SET kept_at = '2026-01-01T00:00:00Z'")
        result_ids = [row["id"] for row in db.list_kept_briefings()]
        assert result_ids == list(reversed(ids))
    finally:
        db.close_connection()


def test_list_kept_briefings_pagination_uses_real_sql_limit_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    try:
        ids = [
            db.create_kept_briefing(
                source_briefing_id=i,
                watchlist_name="W",
                body_markdown="B",
                origin="manual",
            )
            for i in range(5)
        ]
        with db.transaction() as cursor:
            cursor.execute("UPDATE kept_briefings SET kept_at = '2026-01-01T00:00:00Z'")

        calls = []
        real_execute_query = db.execute_query

        def spy(query, params=None, **kwargs):
            calls.append((query, params))
            return real_execute_query(query, params, **kwargs)

        monkeypatch.setattr(db, "execute_query", spy)

        page = db.list_kept_briefings(limit=2, offset=1)

        assert len(calls) == 1
        query, params = calls[0]
        assert "LIMIT ? OFFSET ?" in query
        assert params[-2:] == (2, 1)

        expected_order = list(reversed(ids))  # ties -> id DESC
        assert [row["id"] for row in page] == expected_order[1:3]
    finally:
        db.close_connection()


# --- delete_kept_briefing: hard delete + FK cascade ---------------------


def test_foreign_keys_pragma_is_enabled_for_this_connection(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        value = db.get_connection().execute("PRAGMA foreign_keys").fetchone()[0]
        assert value == 1
    finally:
        db.close_connection()


def test_delete_kept_briefing_returns_false_when_absent(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        assert db.delete_kept_briefing(12345) is False
    finally:
        db.close_connection()


def test_delete_kept_briefing_removes_the_row(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=7,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        assert db.delete_kept_briefing(kept_id) is True
        assert db.get_kept_briefing(kept_id) is None
        assert db.delete_kept_briefing(kept_id) is False
    finally:
        db.close_connection()


def test_delete_kept_briefing_cascades_kept_scripts_by_observation(
    tmp_path: Path,
) -> None:
    """Observe the cascade actually happening (delete + re-read), rather
    than asserting on the DDL text alone."""
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=8,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        db.create_kept_script(
            kept_id,
            source_script_id=100,
            preset_name="duo",
            roster_snapshot_json="{}",
            turns_json="[]",
        )
        db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="duo",
            roster_snapshot_json="{}",
            turns_json="[]",
        )

        conn = db.get_connection()
        before_count = conn.execute(
            "SELECT COUNT(*) FROM kept_scripts WHERE kept_briefing_id = ?",
            (kept_id,),
        ).fetchone()[0]
        assert before_count == 2

        assert db.delete_kept_briefing(kept_id) is True

        after_count = conn.execute(
            "SELECT COUNT(*) FROM kept_scripts WHERE kept_briefing_id = ?",
            (kept_id,),
        ).fetchone()[0]
        assert after_count == 0
        assert db.list_kept_scripts(kept_id) == []
    finally:
        db.close_connection()


# --- create_kept_script / list_kept_scripts / kept_script_source_ids ----


def test_create_kept_script_round_trips_all_fields(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=20,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        script_id = db.create_kept_script(
            kept_id,
            source_script_id=555,
            preset_name="duo-cast",
            roster_snapshot_json='{"roster": ["A", "B"]}',
            turns_json='[{"speaker": "A", "text": "Hi"}]',
            model_used="gpt-test",
            original_created_at="2026-07-30T11:00:00Z",
        )
        assert isinstance(script_id, int)

        rows = db.list_kept_scripts(kept_id)
        assert len(rows) == 1
        row = rows[0]
        assert row["kept_briefing_id"] == kept_id
        assert row["source_script_id"] == 555
        assert row["preset_name"] == "duo-cast"
        assert row["roster_snapshot_json"] == '{"roster": ["A", "B"]}'
        assert row["turns_json"] == '[{"speaker": "A", "text": "Hi"}]'
        assert row["model_used"] == "gpt-test"
        assert row["original_created_at"] == datetime(
            2026, 7, 30, 11, 0, tzinfo=timezone.utc
        )
        assert row["kept_at"]
    finally:
        db.close_connection()


def test_create_kept_script_accepts_explicit_kept_at(tmp_path: Path) -> None:
    """Mirrors `create_kept_briefing`'s explicit `kept_at` param -- the
    chatbook importer (task-1870) needs a kept script's `kept_at` to
    survive a cross-device import verbatim rather than being re-stamped
    (task-1870 fix-wave F4)."""
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=203,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        script_id = db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="p",
            roster_snapshot_json="{}",
            turns_json="[]",
            kept_at="2020-01-02T00:00:00Z",
        )
        rows = db.list_kept_scripts(kept_id)
        row = next(r for r in rows if r["id"] == script_id)
        assert row["kept_at"] == datetime(2020, 1, 2, 0, 0, tzinfo=timezone.utc)
    finally:
        db.close_connection()


def test_get_kept_script_by_source_returns_matching_row(tmp_path: Path) -> None:
    """`get_kept_script_by_source` (task-1870) mirrors
    `get_kept_briefing_by_source` -- used by the chatbook importer to
    classify a `ConflictError` on `create_kept_script` as already-present
    vs. a genuine content conflict."""
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=21,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        db.create_kept_script(
            kept_id,
            source_script_id=777,
            preset_name="duo-cast",
            roster_snapshot_json='{"roster": ["A", "B"]}',
            turns_json='[{"speaker": "A", "text": "Hi"}]',
            model_used="gpt-test",
            original_created_at="2026-07-30T11:00:00Z",
        )

        row = db.get_kept_script_by_source(777)
        assert row is not None
        assert row["kept_briefing_id"] == kept_id
        assert row["preset_name"] == "duo-cast"
        assert row["roster_snapshot_json"] == '{"roster": ["A", "B"]}'
    finally:
        db.close_connection()


def test_get_kept_script_by_source_returns_none_when_absent(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        assert db.get_kept_script_by_source(9999) is None
    finally:
        db.close_connection()


def test_create_kept_script_rejects_nonexistent_kept_briefing(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        with pytest.raises(CharactersRAGDBError):
            db.create_kept_script(
                99999,
                preset_name="duo",
                roster_snapshot_json="{}",
                turns_json="[]",
            )
    finally:
        db.close_connection()


def test_create_kept_script_duplicate_source_script_id_raises_conflict(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=9,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        db.create_kept_script(
            kept_id,
            source_script_id=55,
            preset_name="duo",
            roster_snapshot_json="{}",
            turns_json="[]",
        )
        with pytest.raises(ConflictError):
            db.create_kept_script(
                kept_id,
                source_script_id=55,
                preset_name="trio",
                roster_snapshot_json="{}",
                turns_json="[]",
            )
        assert len(db.list_kept_scripts(kept_id)) == 1
    finally:
        db.close_connection()


def test_create_kept_script_allows_multiple_null_source_scripts(
    tmp_path: Path,
) -> None:
    """SQLite treats NULLs in a UNIQUE column as mutually distinct: two
    scripts cast directly from a kept briefing (no subscriptions-side
    source) must both be insertable."""
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=10,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        script_id_a = db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="duo",
            roster_snapshot_json="{}",
            turns_json="[]",
        )
        script_id_b = db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="trio",
            roster_snapshot_json="{}",
            turns_json="[]",
        )
        assert script_id_a != script_id_b
        rows = db.list_kept_scripts(kept_id)
        assert {row["id"] for row in rows} == {script_id_a, script_id_b}
        assert all(row["source_script_id"] is None for row in rows)
    finally:
        db.close_connection()


def test_list_kept_scripts_orders_and_paginates_with_real_sql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=11,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        ids = [
            db.create_kept_script(
                kept_id,
                source_script_id=None,
                preset_name=f"p{i}",
                roster_snapshot_json="{}",
                turns_json="[]",
            )
            for i in range(4)
        ]
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE kept_scripts SET kept_at = '2026-01-01T00:00:00Z' "
                "WHERE kept_briefing_id = ?",
                (kept_id,),
            )

        calls = []
        real_execute_query = db.execute_query

        def spy(query, params=None, **kwargs):
            calls.append((query, params))
            return real_execute_query(query, params, **kwargs)

        monkeypatch.setattr(db, "execute_query", spy)

        page = db.list_kept_scripts(kept_id, limit=2, offset=1)

        assert calls
        query, params = calls[-1]
        assert "LIMIT ? OFFSET ?" in query
        assert params[-2:] == (2, 1)

        expected_order = list(reversed(ids))  # ties -> id DESC
        assert [row["id"] for row in page] == expected_order[1:3]
    finally:
        db.close_connection()


def test_kept_script_source_ids_excludes_nulls(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        kept_id = db.create_kept_briefing(
            source_briefing_id=12,
            watchlist_name="W",
            body_markdown="B",
            origin="manual",
        )
        db.create_kept_script(
            kept_id,
            source_script_id=None,
            preset_name="p",
            roster_snapshot_json="{}",
            turns_json="[]",
        )
        db.create_kept_script(
            kept_id,
            source_script_id=200,
            preset_name="p",
            roster_snapshot_json="{}",
            turns_json="[]",
        )
        db.create_kept_script(
            kept_id,
            source_script_id=201,
            preset_name="p",
            roster_snapshot_json="{}",
            turns_json="[]",
        )
        assert db.kept_script_source_ids(kept_id) == {200, 201}
    finally:
        db.close_connection()


def test_kept_script_counts_returns_correct_counts_for_multiple_briefings(
    tmp_path: Path,
) -> None:
    """A grouped `COUNT(*)`, not a per-briefing
    `len(list_kept_scripts(...))` -- backs the chatbook creation UI's
    content tree, which used to materialize every kept script's full
    `turns_json`/`roster_snapshot_json` on the UI thread purely to compute
    a subtitle count (task-1870 fix-wave F3)."""
    db = _make_db(tmp_path)
    try:
        two_scripts_id = db.create_kept_briefing(
            source_briefing_id=30,
            watchlist_name="Two Scripts",
            body_markdown="B",
            origin="manual",
        )
        one_script_id = db.create_kept_briefing(
            source_briefing_id=31,
            watchlist_name="One Script",
            body_markdown="B",
            origin="manual",
        )
        zero_scripts_id = db.create_kept_briefing(
            source_briefing_id=32,
            watchlist_name="Zero Scripts",
            body_markdown="B",
            origin="manual",
        )
        for _ in range(2):
            db.create_kept_script(
                two_scripts_id,
                source_script_id=None,
                preset_name="p",
                roster_snapshot_json="{}",
                turns_json="[]",
            )
        db.create_kept_script(
            one_script_id,
            source_script_id=None,
            preset_name="p",
            roster_snapshot_json="{}",
            turns_json="[]",
        )

        counts = db.kept_script_counts(
            [two_scripts_id, one_script_id, zero_scripts_id]
        )

        assert counts == {
            two_scripts_id: 2,
            one_script_id: 1,
            zero_scripts_id: 0,
        }
    finally:
        db.close_connection()


def test_kept_script_counts_omits_scripts_for_ids_not_requested(
    tmp_path: Path,
) -> None:
    """A briefing's scripts must not leak into another briefing's count
    just because both exist in the database -- only requested ids appear,
    each scoped to its own `kept_briefing_id`."""
    db = _make_db(tmp_path)
    try:
        requested_id = db.create_kept_briefing(
            source_briefing_id=33,
            watchlist_name="Requested",
            body_markdown="B",
            origin="manual",
        )
        other_id = db.create_kept_briefing(
            source_briefing_id=34,
            watchlist_name="Not requested",
            body_markdown="B",
            origin="manual",
        )
        db.create_kept_script(
            other_id,
            source_script_id=None,
            preset_name="p",
            roster_snapshot_json="{}",
            turns_json="[]",
        )

        counts = db.kept_script_counts([requested_id])

        assert counts == {requested_id: 0}
        assert other_id not in counts
    finally:
        db.close_connection()


def test_kept_script_counts_returns_zero_for_nonexistent_ids(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        assert db.kept_script_counts([9999]) == {9999: 0}
    finally:
        db.close_connection()


def test_kept_script_counts_returns_empty_dict_for_empty_input(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        assert db.kept_script_counts([]) == {}
    finally:
        db.close_connection()


# --- Migration: fresh vs. upgraded paths reach the same schema ---------


def test_fresh_database_reaches_v29_with_kept_tables(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "fresh.sqlite")
    try:
        connection = db.get_connection()
        # task-1780's kept_briefings/kept_scripts land at v29, but cost
        # ticker PR1 (v29->v30, local-only messages.usage_json) bumped the
        # schema further; a fresh DB always migrates to whatever is
        # current, so assert dynamically rather than re-pinning a version
        # number this file doesn't own.
        assert _version(connection) == db._CURRENT_SCHEMA_VERSION
        assert {"kept_briefings", "kept_scripts"} <= _table_names(connection)
    finally:
        db.close_connection()


def test_v28_database_migrates_to_v29_and_gains_kept_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "migrated.sqlite"
    _seed_v28_database(path, monkeypatch)

    db = open_current_chachanotes_from_legacy(
        path, client_id="kept-migrated"
    )
    try:
        connection = db.get_connection()
        # See the dynamic-assertion note above: cost ticker PR1 bumped the
        # schema past v29.
        assert _version(connection) == db._CURRENT_SCHEMA_VERSION
        assert {"kept_briefings", "kept_scripts"} <= _table_names(connection)
    finally:
        db.close_connection()


def test_fresh_and_migrated_databases_reach_identical_kept_table_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh_path = tmp_path / "fresh-cols.sqlite"
    migrated_path = tmp_path / "migrated-cols.sqlite"
    _seed_v28_database(migrated_path, monkeypatch)

    fresh_db = CharactersRAGDB(fresh_path, client_id="kept-fresh-cols")
    migrated_db = open_current_chachanotes_from_legacy(
        migrated_path, client_id="kept-migrated-cols"
    )
    try:
        for table in ("kept_briefings", "kept_scripts"):
            fresh_shape = [
                tuple(row)
                for row in fresh_db.get_connection()
                .execute(f"PRAGMA table_info({table})")
                .fetchall()
            ]
            migrated_shape = [
                tuple(row)
                for row in migrated_db.get_connection()
                .execute(f"PRAGMA table_info({table})")
                .fetchall()
            ]
            assert fresh_shape == migrated_shape
    finally:
        fresh_db.close_connection()
        migrated_db.close_connection()


def test_migration_sql_kept_tables_have_no_sync_columns_or_fts() -> None:
    executable = "\n".join(
        line
        for line in MIGRATION_SQL_PATH.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    ).lower()

    assert "create virtual table" not in executable
    assert "create trigger" not in executable

    briefings_body = _extract_table_body(executable, "kept_briefings")
    scripts_body = _extract_table_body(executable, "kept_scripts")
    for body in (briefings_body, scripts_body):
        assert "client_id" not in body
        assert "version" not in body
        assert "deleted" not in body

    assert "on delete cascade" in scripts_body
    assert "on delete cascade" not in briefings_body


def test_inline_migration_sql_matches_migration_file() -> None:
    """Guard against the runner SQL constant and the on-disk migration file
    (kept side by side for readability -- see either's header comment)
    drifting apart."""
    file_sql = MIGRATION_SQL_PATH.read_text(encoding="utf-8")
    file_ddl = file_sql[file_sql.index("CREATE TABLE") :]
    assert CharactersRAGDB._MIGRATE_V28_TO_V29_SQL.strip() == file_ddl.strip()
