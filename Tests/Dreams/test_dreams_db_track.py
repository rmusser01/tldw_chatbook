# Tests/Dreams/test_dreams_db_track.py
"""DreamsDB schema v2 behavior: tracked items, track runs, v1->v2 upgrade.

Also pins the query plans of the three track-loop indexes (the
``scripts/check_index_plan_pins.py`` census rows point here): every plan
assertion first proves ``sqlite_stat1`` is absent, because Dreams_DB.py
runs no ANALYZE and a plan captured with stats is not the plan a user's
database produces.
"""
import sqlite3

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB


@pytest.fixture()
def db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "test-client")
    yield database
    database.close()


def _make_item(db, **overrides):
    kwargs: dict = {
        "mechanism": "question",
        "intent": "event",
        "cadence_seconds": 3600,
    }
    kwargs.update(overrides)
    return db.create_tracked_item(**kwargs)


def test_create_and_get_tracked_item_roundtrip_with_defaults(db):
    item_id = _make_item(
        db,
        origin_story_id=42,
        subscription_id=7,
        query_template="flights to {place}",
        event_date="2026-10-01",
    )
    item = db.get_tracked_item(item_id)
    assert item is not None
    assert item["id"] == item_id
    assert item["origin_story_id"] == 42
    assert item["mechanism"] == "question"
    assert item["intent"] == "event"
    assert item["subscription_id"] == 7
    assert item["query_template"] == "flights to {place}"
    assert item["event_date"] == "2026-10-01"
    assert item["cadence_seconds"] == 3600
    # Column defaults from the DDL
    assert item["quiet_retire_count"] == 0
    assert item["status"] == "active"
    assert item["retired_reason"] is None
    assert item["created_by_dreams"] == 0
    assert item["last_checked"] is None
    assert item["created_at"]
    assert item["updated_at"]


def test_get_tracked_item_missing_returns_none(db):
    assert db.get_tracked_item(999) is None


def test_create_tracked_item_rejects_mechanism_outside_vocabulary(db):
    with pytest.raises(sqlite3.IntegrityError):
        _make_item(db, mechanism="bogus")


def test_create_tracked_item_rejects_intent_outside_vocabulary(db):
    with pytest.raises(sqlite3.IntegrityError):
        _make_item(db, intent="bogus")


def test_list_tracked_items_orders_newest_first_and_filters_status(db):
    first = _make_item(db, query_template="q1")
    second = _make_item(db, query_template="q2")
    third = _make_item(db, query_template="q3")
    db.set_tracked_status(second, "paused")
    active = db.list_tracked_items()
    assert [i["id"] for i in active] == [third, first]
    assert [i["id"] for i in db.list_tracked_items(status="paused")] == [second]
    assert [i["id"] for i in db.list_tracked_items(status="retired")] == []


def test_find_tracked_by_story(db):
    assert db.find_tracked_by_story(42) is None
    item_id = _make_item(db, origin_story_id=42)
    found = db.find_tracked_by_story(42)
    assert found is not None
    assert found["id"] == item_id


def test_find_tracked_by_story_returns_only_active_rows(db):
    """Final review, ruling P10: a retired/paused wrapper is not untrackable.

    Without the status filter the story modal's ``u`` resolves a
    sweep-retired row, re-retires it (``retired_reason`` COALESCE-rewrites
    the sweep's ``event_passed``/``quiet`` audit data), re-disables the
    already-disabled subscription, and posts "Stopped tracking this page."
    for a watch the user never manually stopped.
    """
    retired = _make_item(db, origin_story_id=42)
    db.set_tracked_status(retired, "retired", retired_reason="event_passed")
    assert db.find_tracked_by_story(42) is None, (
        "a retired row must not surface for untrack"
    )

    paused = _make_item(db, origin_story_id=42)
    db.set_tracked_status(paused, "paused")
    assert db.find_tracked_by_story(42) is None, (
        "a paused row must not surface for untrack either"
    )

    active = _make_item(db, origin_story_id=42)
    assert db.find_tracked_by_story(42)["id"] == active


def test_set_tracked_status_stamps_retired_reason_and_updated_at(db):
    item_id = _make_item(db)
    # Backdate the stamp so the update must land in a strictly later
    # millisecond (creation and update can tie within one ms).
    with db.transaction() as conn:
        conn.execute(
            "UPDATE dream_tracked_items SET updated_at = '2020-01-01T00:00:00.000Z'"
            " WHERE id = ?",
            (item_id,),
        )
    db.set_tracked_status(item_id, "retired", retired_reason="quiet: too many duds")
    after = db.get_tracked_item(item_id)
    assert after["status"] == "retired"
    assert after["retired_reason"] == "quiet: too many duds"
    assert after["updated_at"] > "2020-01-01T00:00:00.000Z"


def test_set_tracked_status_rejects_status_outside_vocabulary(db):
    item_id = _make_item(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.set_tracked_status(item_id, "bogus")


def test_touch_tracked_checked_stamps_last_checked(db):
    item_id = _make_item(db)
    db.touch_tracked_checked(item_id, "2026-09-28T12:00:00.000Z")
    item = db.get_tracked_item(item_id)
    assert item["last_checked"] == "2026-09-28T12:00:00.000Z"
    assert item["updated_at"] >= "2026-09-28T12:00:00.000Z"


def test_count_active_tracked_counts_only_active(db):
    a = _make_item(db)
    b = _make_item(db)
    c = _make_item(db)
    assert db.count_active_tracked() == 3
    db.set_tracked_status(b, "paused")
    db.set_tracked_status(c, "retired")
    assert db.count_active_tracked() == 1
    assert db.list_tracked_items()[0]["id"] == a


def test_insert_and_list_recent_track_runs_newest_first(db):
    item_id = _make_item(db)
    db.insert_track_run(item_id, status="baseline", digest_hash="h0")
    db.insert_track_run(item_id, status="unchanged", digest_hash="h1")
    db.insert_track_run(item_id, status="changed", digest_hash="h2",
                        verdict_note="price dropped", notified=1)
    runs = db.list_recent_track_runs(item_id)
    assert [r["status"] for r in runs] == ["changed", "unchanged", "baseline"]
    newest = runs[0]
    assert newest["digest_hash"] == "h2"
    assert newest["verdict_note"] == "price dropped"
    assert newest["notified"] == 1
    # Limit slices the newest end, not the oldest
    assert [r["status"] for r in db.list_recent_track_runs(item_id, limit=2)] == \
        ["changed", "unchanged"]
    assert db.list_recent_track_runs(999) == []


def test_insert_track_run_rejects_status_outside_vocabulary(db):
    item_id = _make_item(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_track_run(item_id, status="bogus", digest_hash="h")


def test_consecutive_track_dispositions_counts_trailing_run(db):
    item_id = _make_item(db)
    # Insert order doubles as age order: baseline, unchanged, unchanged, changed
    for status in ("baseline", "unchanged", "unchanged", "changed"):
        db.insert_track_run(item_id, status=status, digest_hash="h")
    # Newest run is 'changed', so only 'changed' has a trailing run
    assert db.consecutive_track_dispositions(item_id, "changed") == 1
    assert db.consecutive_track_dispositions(item_id, "unchanged") == 0
    assert db.consecutive_track_dispositions(item_id, "baseline") == 0

    other_id = _make_item(db)
    for _ in range(3):
        db.insert_track_run(other_id, status="unchanged", digest_hash="h")
    assert db.consecutive_track_dispositions(other_id, "unchanged") == 3
    # No runs at all -> zero for any status
    empty_id = _make_item(db)
    assert db.consecutive_track_dispositions(empty_id, "unchanged") == 0


def _rewind_to_v1(db) -> None:
    """Rewind an open v2 database file to a faithful v1 state.

    v2 is additive only, so dropping the two track tables (and their indexes)
    and re-stamping the version row leaves exactly what a Phase 1 writer
    would have on disk.
    """
    with db.transaction() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_dream_track_runs_item")
        conn.execute("DROP INDEX IF EXISTS idx_dream_tracked_status")
        conn.execute("DROP INDEX IF EXISTS idx_dream_tracked_origin_story")
        conn.execute("DROP TABLE IF EXISTS dream_track_runs")
        conn.execute("DROP TABLE IF EXISTS dream_tracked_items")
        # A v1-created file holds exactly one stamp: version 1 (a fresh v2
        # file only ever writes version 2, so rewriting the table, not
        # deleting one row, is the faithful old-file shape).
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")


def test_v1_file_upgrades_in_place_to_v2(tmp_path):
    path = tmp_path / "dreams-v1.sqlite"
    db = DreamsDB(path, "test-client")
    db.create_collection("2026-09-22", "scheduled", "digest")
    _rewind_to_v1(db)
    with db.connection() as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name = 'dream_tracked_items'"
        ).fetchone() is None
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
    db.close()

    # Reopening the v1 file must upgrade it in place via the additive DDL
    # (through every later additive bump -- v3 adds the 'guardian' profile
    # source CHECK, so a v1 file lands on version 3 as well).
    reopened = DreamsDB(path, "test-client")
    with reopened.connection() as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        indexes = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_dream_%'"
            ).fetchall()
        }
        assert "dream_tracked_items" in tables
        assert "dream_track_runs" in tables
        assert "idx_dream_tracked_status" in indexes
        assert "idx_dream_track_runs_item" in indexes
        # Task 6 ledger item: the per-story reads Task 5 added
        # (``find_tracked_by_story`` + the badge) get their pinned index.
        assert "idx_dream_tracked_origin_story" in indexes
        assert conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == DreamsDB._CURRENT_SCHEMA_VERSION
    # The upgraded file is fully usable: track CRUD works and v1 data survived.
    item_id = reopened.create_tracked_item(
        mechanism="page", intent="deal", cadence_seconds=60
    )
    reopened.insert_track_run(item_id, status="baseline", digest_hash="h")
    assert reopened.count_active_tracked() == 1
    assert reopened.get_collection_by_date("2026-09-22")["status"] == "generating"
    reopened.close()


# --- Schema v3 (Guardian x Dreams, ADR-204 contract 3) -------------------------


_PROFILE_V2_DDL = """
CREATE TABLE dream_interest_profile (
    id INTEGER PRIMARY KEY,
    facet TEXT NOT NULL CHECK(facet IN ('topic', 'goal')),
    text TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    searchable INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL
        CHECK(source IN ('user', 'seed', 'personal_context', 'notes', 'media')),
    query_angle TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_boosted_at TEXT,
    UNIQUE(facet, text)
)
"""


def _rewind_to_v2(db) -> None:
    """Rewind an open v3 database file to a faithful v2 state.

    v3 changes a CHECK constraint (``source`` gains ``'guardian'``), and
    SQLite CHECKs are baked into the table DDL, so the faithful rewind
    rebuilds ``dream_interest_profile`` with the v2 shape. A real v2 writer
    could never have stored ``source='guardian'`` rows (the CHECK refuses
    them), so such rows are dropped, not carried. Re-stamping the version
    row to 2 completes the old-file shape.
    """
    with db.transaction() as conn:
        conn.execute("DROP TABLE IF EXISTS dream_interest_profile_v2")
        conn.execute(
            _PROFILE_V2_DDL.replace(
                "dream_interest_profile", "dream_interest_profile_v2", 1
            )
        )
        conn.execute(
            "INSERT INTO dream_interest_profile_v2"
            " SELECT * FROM dream_interest_profile"
            " WHERE source != 'guardian'"
        )
        conn.execute("DROP TABLE dream_interest_profile")
        conn.execute(
            "ALTER TABLE dream_interest_profile_v2"
            " RENAME TO dream_interest_profile"
        )
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (2)")


def test_v2_file_upgrades_in_place_to_v3_and_accepts_guardian_source(tmp_path):
    path = tmp_path / "dreams-v2.sqlite"
    db = DreamsDB(path, "test-client")
    db.upsert_profile_entry(
        "topic", "rust tui", weight=0.8, searchable=1, source="notes"
    )
    _rewind_to_v2(db)
    with db.connection() as conn:
        assert conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == 2
        # The v2 CHECK refuses the guardian source on the rewound file.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO dream_interest_profile"
                " (facet, text, weight, searchable, source, created_at,"
                "  updated_at) VALUES ('topic', 'x', 0.5, 1, 'guardian',"
                " '2026-09-29T00:00:00Z', '2026-09-29T00:00:00Z')"
            )
    db.close()

    # Reopening the v2 file must upgrade it in place: version stamps to 3
    # and the rebuilt CHECK accepts 'guardian' through the public writer.
    reopened = DreamsDB(path, "test-client")
    assert reopened._CURRENT_SCHEMA_VERSION == 3
    reopened.upsert_profile_entry(
        "topic", "late_night_work", weight=0.75, searchable=1, source="guardian"
    )
    with reopened.connection() as conn:
        assert conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == 3
    by_text = {row["text"]: row for row in reopened.list_profile()}
    assert by_text["rust tui"]["source"] == "notes", "v2 data survived"
    assert by_text["late_night_work"]["source"] == "guardian"
    reopened.close()


def test_fresh_v3_build_never_runs_the_profile_rebuild_migration(tmp_path):
    """A fresh build creates the new-CHECK table directly; the v2 -> v3
    rebuild migration must be a no-op there (pinned by table id continuity:
    the rebuild would have replaced the table object)."""
    db = DreamsDB(tmp_path / "dreams-fresh.sqlite", "test-client")
    db.upsert_profile_entry(
        "topic", "seeded", weight=1.0, searchable=1, source="seed"
    )
    with db.connection() as conn:
        (profile_id,) = conn.execute(
            "SELECT id FROM dream_interest_profile"
        ).fetchone()
        assert conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == 3
    assert profile_id == 1, "the seeded row keeps the first autoincrement id"
    db.close()


# --- Qodo review on PR #2890 -----------------------------------------------------


def test_list_tracked_items_is_bounded(db):
    """Qodo #9: retired rows accumulate forever, so the status read is
    capped (TRACKED_ITEMS_LIST_LIMIT, mirroring ``clamp_limit``'s spirit);
    the default cap is respected, callers can page below it, and garbage
    limits clamp instead of unbounding (SQLite reads negative LIMIT as
    no-limit)."""
    from tldw_chatbook.DB.Dreams_DB import TRACKED_ITEMS_LIST_LIMIT

    seeded = TRACKED_ITEMS_LIST_LIMIT + 5
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO dream_tracked_items"
            " (mechanism, intent, cadence_seconds, status, query_template,"
            "  created_at, updated_at)"
            " VALUES ('question', 'topic', 3600, 'retired', ?, ?, ?)",
            [(f"q {i}", "2026-09-01T00:00:00+00:00",
              "2026-09-01T00:00:00+00:00") for i in range(seeded)],
        )

    assert len(db.list_tracked_items("retired")) == TRACKED_ITEMS_LIST_LIMIT
    assert len(db.list_tracked_items("retired", limit=10)) == 10
    assert len(db.list_tracked_items("retired", limit=-5)) == 1, (
        "a negative limit clamps to 1, the way ``clamp_limit`` does"
    )


def test_set_tracked_plan_rewrites_cadence_and_event_date(db):
    """Qodo #2 support: the reuse path refreshes the caller's plan on the
    existing active wrapper -- both fields, including clearing the date."""
    item = db.create_tracked_item(
        mechanism="page", intent="event", cadence_seconds=43200,
        event_date="2026-10-01", subscription_id=9, created_by_dreams=1)

    db.set_tracked_plan(item, cadence_seconds=86400, event_date=None)

    row = db.get_tracked_item(item)
    assert row["cadence_seconds"] == 86400
    assert row["event_date"] is None
    assert row["status"] == "active"


# --- Query-plan pins (scripts/check_index_plan_pins.py census rows) -------------
#
# TASK-21126's rule: with no sqlite_stat1 the planner may ignore a perfect
# index, so "the index exists" proves nothing. Each pin below captures
# EXPLAIN QUERY PLAN for the ACTUAL production SQL (kept as literals on
# purpose -- importing the reader's string would hide a reader/index
# mismatch) on a seeded database with sqlite_stat1 ABSENT, and asserts the
# census-recorded index is the one the planner seeks.
#
# Honest scope, measured rather than assumed: unlike the media v9 set, these
# indexes buy the equality SEEK, not the sort. ``created_at DESC, id DESC``
# still goes through a temp B-tree (the indexes do not carry the ordering
# columns), and the ``id DESC`` tiebreak term keeps one even where the
# leading ordering column is indexed. The pins below assert the SEEK and the
# flip from SCAN (see the negative control); they deliberately claim
# nothing about sort elimination.

#: The three census rows in scripts/index_plan_pin_census.tsv.
RUNS_ITEM = "idx_dream_track_runs_item"
TRACKED_STATUS = "idx_dream_tracked_status"
TRACKED_ORIGIN_STORY = "idx_dream_tracked_origin_story"

#: Production SQL exactly as the readers spell it, with the binds each
#: caller actually passes.
RUNS_LIST_SQL = (
    "SELECT * FROM dream_track_runs WHERE tracked_item_id = ?"
    " ORDER BY created_at DESC, id DESC LIMIT ?"
)
TRACKED_LIST_SQL = (
    "SELECT * FROM dream_tracked_items WHERE status = ?"
    " ORDER BY created_at DESC, id DESC LIMIT ?"
)
ACTIVE_COUNT_SQL = (
    "SELECT COUNT(*) FROM dream_tracked_items WHERE status = 'active'"
)
STORY_LOOKUP_SQL = (
    "SELECT * FROM dream_tracked_items WHERE origin_story_id = ?"
    " AND status = ?"
    " ORDER BY created_at DESC, id DESC LIMIT 1"
)

PLAN_PINNED_QUERIES = {
    "list_recent_track_runs": (RUNS_LIST_SQL, (5, 5), RUNS_ITEM),
    "list_tracked_items": (TRACKED_LIST_SQL, ("active", 500), TRACKED_STATUS),
    "count_active_tracked": (ACTIVE_COUNT_SQL, (), TRACKED_STATUS),
    "find_tracked_by_story": (STORY_LOOKUP_SQL, (7, "active"), TRACKED_ORIGIN_STORY),
}

CONSECUTIVE_DISPOSITIONS_SQL = (
    "SELECT COUNT(*) FROM dream_track_runs AS r"
    " WHERE r.tracked_item_id = ? AND r.status = ?"
    " AND r.id > COALESCE(("
    "     SELECT b.id FROM dream_track_runs AS b"
    "     WHERE b.tracked_item_id = ? AND b.status <> ?"
    "     ORDER BY b.created_at DESC, b.id DESC LIMIT 1"
    " ), 0)"
)


def _assert_no_stats(conn) -> None:
    """Prove the fixture reproduces the no-stats production state.

    Dreams_DB.py runs no ANALYZE, so no user's database carries
    ``sqlite_stat1`` and a plan captured with one present is not the plan
    they run.
    """
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'"
    ).fetchone() is None


def _plan(conn, sql: str, params: tuple = ()) -> str:
    """Capture EXPLAIN QUERY PLAN, stats-absent, as one detail string."""
    _assert_no_stats(conn)
    detail = " | ".join(
        row["detail"] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params)
    )
    assert detail, "an empty plan satisfies every negative assertion"
    return detail


def _seed_plan_corpus(db) -> None:
    """Insert tracked items and runs directly (fast, shape-exact).

    Goes around the writers deliberately: this section cares about the
    physical row shape the planner sees, and needs all three lifecycle
    statuses and NULL/duplicate origin stories in one table.
    300 items (100 per status, half with an origin story in 1..50) and
    600 runs over 12 items -- enough rows that the stats-free planner's
    choices are the ones it makes on a real database, not a toy.
    """
    items = []
    for i in range(1, 301):
        status = "active" if i % 3 == 0 else ("paused" if i % 3 == 1 else "retired")
        items.append(
            (
                i,
                (i % 50) + 1 if i % 2 else None,
                "question",
                "event",
                3600,
                status,
                f"2026-08-{1 + i % 28:02d}T00:00:00Z",
                "2026-08-01T00:00:00Z",
            )
        )
    runs = [
        (
            item,
            "unchanged" if j % 5 else "changed",
            f"h{j}",
            f"2026-09-{1 + j % 28:02d}T00:00:{j % 60:02d}Z",
        )
        for item in range(1, 13)
        for j in range(50)
    ]
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO dream_tracked_items (id, origin_story_id, mechanism,"
            " intent, cadence_seconds, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            items,
        )
        conn.executemany(
            "INSERT INTO dream_track_runs (tracked_item_id, status, digest_hash,"
            " created_at) VALUES (?,?,?,?)",
            runs,
        )


@pytest.mark.parametrize("label", sorted(PLAN_PINNED_QUERIES))
def test_production_query_plan_seeks_its_census_index(db, label):
    sql, params, expected = PLAN_PINNED_QUERIES[label]
    _seed_plan_corpus(db)
    with db.connection() as conn:
        plan = _plan(conn, sql, params)
    assert expected in plan, f"{label}: {plan}"
    # count_active_tracked additionally reads the index alone (COVERING):
    # no table row is touched for the cap guard's COUNT.
    if label == "count_active_tracked":
        assert f"COVERING INDEX {TRACKED_STATUS}" in plan, plan


def test_consecutive_track_dispositions_plan_seeks_the_runs_index_both_halves(db):
    """The quiet-retire gate's COUNT and its scalar subquery both seek.

    The outer ``r`` scan and the inner ``b`` lookup each filter on
    ``tracked_item_id`` alone, so both must land on
    ``idx_dream_track_runs_item`` -- the subquery is the one that runs per
    retirement decision, and a SCAN there is the slow path this index
    exists to remove.
    """
    _seed_plan_corpus(db)
    with db.connection() as conn:
        plan = _plan(
            conn,
            CONSECUTIVE_DISPOSITIONS_SQL,
            (5, "unchanged", 5, "unchanged"),
        )
    assert f"SEARCH r USING INDEX {RUNS_ITEM}" in plan, plan
    assert f"SEARCH b USING INDEX {RUNS_ITEM}" in plan, plan


def test_dropping_the_indexes_flips_every_pinned_query_to_a_scan(db):
    """The negative control: each pin means the planner COULD have refused.

    Without the three indexes every pinned read loses its seek (captured
    with sqlite_stat1 absent, like the pins): the four single-table reads
    become full SCANs of their table, and the quiet-retire query's
    per-retirement subquery walks every run while the outer COUNT falls
    back to the rowid PK (``r.id > ?`` stays a rowid range). That is the
    plan family the indexes remove, and what makes "the index name is in
    the plan" a load-bearing assertion rather than a formality.
    """
    _seed_plan_corpus(db)
    with db.connection() as conn:
        for name in (RUNS_ITEM, TRACKED_STATUS, TRACKED_ORIGIN_STORY):
            conn.execute(f"DROP INDEX {name}")
        conn.commit()
        for label, (sql, params, _expected) in PLAN_PINNED_QUERIES.items():
            table = (
                "dream_track_runs"
                if "dream_track_runs" in sql
                else "dream_tracked_items"
            )
            assert f"SCAN {table}" in _plan(conn, sql, params), (
                f"{label}: expected a full scan without the indexes"
            )
        consecutive = _plan(
            conn, CONSECUTIVE_DISPOSITIONS_SQL, (5, "unchanged", 5, "unchanged")
        )
        assert "SCAN b" in consecutive, consecutive
        assert RUNS_ITEM not in consecutive, consecutive
