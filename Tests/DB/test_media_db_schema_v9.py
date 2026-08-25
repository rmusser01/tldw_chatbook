"""Schema v9 (TASK-21593): the four active-media partial indexes, plus the
two ``search_media_db`` rewrites the same audit produced.

TASK-21126 proved for ONE query that an index the stats-free planner will
not choose is a dead index. This file is the rest of that audit. Every
Media list surface -- Library page, Media browse, ``get_paginated_files``,
the selection dropdown, the read-it-later list, the type facet, and all
four sort orders -- filtered ``deleted = 0 AND is_trash = 0`` and then made
SQLite sort the whole live library in a temp B-tree to return twenty rows.

What these tests pin, and why each one exists:

* the four indexes are created on a fresh DB and by a genuine v8 upgrade,
  with the exact measured column order and partial predicate;
* the migration adds NO row, column, table or trigger (a pure index add);
* **the planner actually CHOOSES each of them with no ``sqlite_stat1``**,
  and no list query is left on a temp-B-tree sort. This is the load-bearing
  set: an "index exists" assertion passes for an index nothing uses, and
  the bare ``(last_modified DESC, id DESC) WHERE deleted = 0 AND is_trash =
  0`` shape -- the textbook one for these ORDER BYs -- is measurably never
  picked in the state real databases are in;
* the ``must_have_keywords`` / ``must_not_have_keywords`` subqueries reach
  ``Keywords`` by its unique index rather than walking every live keyword
  per candidate row, and return byte-identical results to the ``LOWER()``
  spelling for every value the caller can bind;
* the FTS COUNT reaches ``media_fts`` first, EXCEPT when an id allowlist
  makes Media the genuinely cheap side.

Every plan assertion here first asserts ``sqlite_stat1`` is absent, so a
future fixture that runs ``ANALYZE`` cannot quietly restore a flattering
plan that no user's database produces.
"""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

from Tests.DB.historical_bootstrap_v6 import media_db_at_version

RECENT = "idx_media_active_recent"
TYPE = "idx_media_active_type"
INGESTED = "idx_media_active_ingested"
TITLE = "idx_media_active_title"
V9_INDEXES = (RECENT, TYPE, INGESTED, TITLE)

PREVIEW = 241
BROAD = (
    "m.id, m.uuid, m.url, m.title, m.type, m.author, m.ingestion_date, "
    "m.transcription_model, m.transcription_provenance_json, m.is_trash, "
    "m.trash_date, m.chunking_status, m.vector_processing, m.content_hash, "
    "m.last_modified, m.version, m.client_id, m.deleted"
)

#: The list SQL exactly as the production readers spell it. Kept as literals
#: on purpose (the v8 file's rule): this file's job is to fail when a reader
#: and its index stop matching, which importing the reader's string hides.
LIST_QUERIES = {
    "list_library_media_page rows": (
        "SELECT id, uuid, title, type, author, ingestion_date, last_modified, "
        "version, substr(content, 1, ?) AS preview FROM Media "
        "WHERE deleted = 0 AND is_trash = 0 "
        "ORDER BY last_modified DESC, id DESC LIMIT ? OFFSET ?",
        (PREVIEW, 25, 0),
        RECENT,
    ),
    "list_library_media_page count": (
        "SELECT COUNT(*) AS count FROM Media WHERE deleted = 0 AND is_trash = 0",
        (),
        None,  # any of the four may serve a bare count; only "not a table scan"
    ),
    "search_media_db browse rows": (
        f"SELECT DISTINCT {BROAD} FROM Media m WHERE m.deleted = 0 AND m.is_trash = 0 "
        f"ORDER BY m.last_modified DESC, m.id DESC LIMIT ? OFFSET ?",
        (20, 0),
        RECENT,
    ),
    "search_media_db sort=date_desc": (
        f"SELECT DISTINCT {BROAD} FROM Media m WHERE m.deleted = 0 AND m.is_trash = 0 "
        f"ORDER BY m.ingestion_date DESC, m.last_modified DESC, m.id DESC "
        f"LIMIT ? OFFSET ?",
        (20, 0),
        INGESTED,
    ),
    "search_media_db sort=title_asc": (
        f"SELECT DISTINCT {BROAD} FROM Media m WHERE m.deleted = 0 AND m.is_trash = 0 "
        f"ORDER BY m.title COLLATE NOCASE ASC, m.id ASC LIMIT ? OFFSET ?",
        (20, 0),
        TITLE,
    ),
    "search_media_db sort=title_desc": (
        f"SELECT DISTINCT {BROAD} FROM Media m WHERE m.deleted = 0 AND m.is_trash = 0 "
        f"ORDER BY m.title COLLATE NOCASE DESC, m.id DESC LIMIT ? OFFSET ?",
        (20, 0),
        TITLE,
    ),
    "get_paginated_files rows": (
        "SELECT id, title, type, last_modified FROM Media "
        "WHERE deleted = 0 AND is_trash = 0 "
        "ORDER BY last_modified DESC, id DESC LIMIT ? OFFSET ?",
        (50, 0),
        RECENT,
    ),
    "get_distinct_media_types": (
        "SELECT DISTINCT type FROM Media "
        "WHERE type IS NOT NULL AND type != '' AND deleted = 0 AND is_trash = 0 "
        "ORDER BY type ASC",
        (),
        TYPE,
    ),
}

#: Queries whose ORDER BY the v9 set removes the temp B-tree from entirely.
#: `get_distinct_media_types` is listed separately because what it loses is
#: the temp B-tree for DISTINCT, not for ORDER BY.
NO_TEMP_SORT = [
    "list_library_media_page rows",
    "search_media_db browse rows",
    "search_media_db sort=title_asc",
    "search_media_db sort=title_desc",
    "get_paginated_files rows",
    "get_distinct_media_types",
]


@pytest.fixture()
def fresh_db(tmp_path):
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="test")
    yield db
    db.close_connection()


def _indexes_on_media(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        row["name"]: row["sql"] or ""
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'Media'"
        )
    }


def _assert_no_stats(conn: sqlite3.Connection) -> None:
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'"
    ).fetchone() is None, (
        "this fixture must reproduce the no-stats production state; "
        "Client_Media_DB_v2.py runs no ANALYZE, so no user's media DB has "
        "sqlite_stat1 and a plan captured with one is not the plan they run"
    )


def _plan(conn: sqlite3.Connection, sql: str, params=()) -> str:
    _assert_no_stats(conn)
    detail = " | ".join(
        row["detail"] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params)
    )
    assert detail, "an empty plan satisfies every negative assertion"
    return detail


def _seed(conn: sqlite3.Connection, *, live: int = 400, trashed: int = 30,
          deleted: int = 30, keywords: int = 40) -> None:
    """Insert Media/Keywords/MediaKeywords rows directly (fast, shape-exact).

    Goes around ``add_media_with_keywords`` deliberately: this file cares
    about the physical row shape the planner sees, and needs trashed and
    soft-deleted rows in the same table, which the writer will not produce
    on demand.
    """
    types = ["video", "audio", "pdf", "document", "web_page"]
    rows = []
    total = live + trashed + deleted
    for i in range(1, total + 1):
        if i <= live:
            is_trash, is_deleted = 0, 0
        elif i <= live + trashed:
            is_trash, is_deleted = 1, 0
        else:
            is_trash, is_deleted = 0, 1
        rows.append(
            (
                i,
                f"https://example.test/{i}",
                f"Title {i % 7}{i}",
                types[i % len(types)],
                f"body text {i} dragon lore",
                f"Author {i % 11}",
                f"2024-{1 + (i % 12):02d}-{1 + (i % 28):02d}T00:00:00Z",
                f"2025-{1 + (i % 12):02d}-{1 + (i % 28):02d}T00:00:00Z",
                is_trash,
                f"hash-{i}",
                f"uuid-{i}",
                is_deleted,
            )
        )
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executemany(
        "INSERT INTO Media (id, url, title, type, content, author, ingestion_date, "
        "last_modified, is_trash, content_hash, uuid, deleted, version, client_id, "
        "chunking_status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,'test','pending')",
        rows,
    )
    conn.executemany(
        "INSERT INTO Keywords (id, keyword, uuid, last_modified, version, client_id, "
        "deleted) VALUES (?,?,?,'2024-05-01T00:00:00Z',1,'test',0)",
        [(k, f"kw{k}", f"kw-uuid-{k}") for k in range(1, keywords + 1)],
    )
    conn.executemany(
        "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?,?)",
        [(i, 1 + (i % keywords)) for i in range(1, total + 1)],
    )
    conn.commit()
    conn.execute("INSERT INTO media_fts(media_fts) VALUES('rebuild')")
    conn.commit()


# ---------------------------------------------------------------------------
# The indexes exist, on a fresh DB and after a genuine upgrade
# ---------------------------------------------------------------------------


def test_fresh_db_is_at_the_current_version(fresh_db):
    version = fresh_db.get_connection().execute(
        "SELECT version FROM schema_version LIMIT 1"
    ).fetchone()["version"]
    assert version == MediaDatabase._CURRENT_SCHEMA_VERSION == 9


@pytest.mark.parametrize(
    "name, columns",
    [
        (RECENT, "(deleted, is_trash, last_modified desc, id desc)"),
        (TYPE, "(deleted, is_trash, type)"),
        (INGESTED, "(deleted, is_trash, ingestion_date desc, id desc)"),
        (TITLE, "(deleted, is_trash, title collate nocase, id)"),
    ],
)
def test_fresh_db_has_each_index_with_its_measured_shape(fresh_db, name, columns):
    indexes = _indexes_on_media(fresh_db.get_connection())
    assert name in indexes, sorted(indexes)
    ddl = " ".join(indexes[name].split()).lower()
    # Column ORDER is the measured part -- see the module docstring and the
    # comment on _ACTIVE_MEDIA_INDEX_MIGRATION_SQL.
    assert columns in ddl, ddl
    assert "where deleted = 0 and is_trash = 0" in ddl, ddl


def test_genuine_v8_db_upgrades_and_gains_the_indexes(tmp_path):
    path = tmp_path / "v8.db"
    with media_db_at_version(path, 8) as old:
        conn = old.get_connection()
        assert conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()["version"] == 8
        assert not set(V9_INDEXES) & set(_indexes_on_media(conn))
        _seed(conn, live=20, trashed=2, deleted=2, keywords=5)

    upgraded = MediaDatabase(str(path), client_id="upgrade")
    try:
        conn = upgraded.get_connection()
        assert conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()["version"] == MediaDatabase._CURRENT_SCHEMA_VERSION
        assert set(V9_INDEXES) <= set(_indexes_on_media(conn))
        # The rows the v8 DB already held are untouched by an index add.
        assert conn.execute("SELECT COUNT(*) AS n FROM Media").fetchone()["n"] == 24
        rows, total = upgraded.search_media_db(search_query=None, results_per_page=5)
        assert total == 20
        assert len(rows) == 5
    finally:
        upgraded.close_connection()


def test_v8_to_v9_adds_nothing_but_the_indexes(tmp_path):
    """A pure index add: no column, table, trigger or row may move."""
    path = tmp_path / "v8-shape.db"

    def _shape(conn: sqlite3.Connection) -> dict[str, set]:
        return {
            "tables": {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            },
            "triggers": {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            },
            "media_columns": {
                r["name"] for r in conn.execute("PRAGMA table_info(Media)")
            },
        }

    with media_db_at_version(path, 8) as old:
        conn = old.get_connection()
        _seed(conn, live=10, trashed=2, deleted=2, keywords=4)
        before = _shape(conn)
        before_indexes = set(_indexes_on_media(conn))
        before_rows = [
            tuple(r)
            for r in conn.execute(
                "SELECT id, title, type, last_modified, is_trash, deleted, version "
                "FROM Media ORDER BY id"
            )
        ]

    upgraded = MediaDatabase(str(path), client_id="upgrade")
    try:
        conn = upgraded.get_connection()
        after_rows = [
            tuple(r)
            for r in conn.execute(
                "SELECT id, title, type, last_modified, is_trash, deleted, version "
                "FROM Media ORDER BY id"
            )
        ]
        assert _shape(conn) == before
        assert after_rows == before_rows
        assert set(_indexes_on_media(conn)) - before_indexes == set(V9_INDEXES)
    finally:
        upgraded.close_connection()


def test_failed_v8_to_v9_rolls_back_and_leaves_a_working_v8_db(tmp_path):
    """A migration that dies mid-script must leave a usable DB on v8."""
    path = tmp_path / "v8-fail.db"
    with media_db_at_version(path, 8) as old:
        _seed(old.get_connection(), live=8, trashed=1, deleted=1, keywords=3)

    broken = MediaDatabase._ACTIVE_MEDIA_INDEX_MIGRATION_SQL.replace(
        "ON Media(deleted, is_trash, ingestion_date DESC, id DESC)",
        "ON Media(no_such_column)",
    )
    from unittest.mock import patch

    from tldw_chatbook.DB.Client_Media_DB_v2 import DatabaseError

    with patch.object(
        MediaDatabase, "_ACTIVE_MEDIA_INDEX_MIGRATION_SQL", broken
    ):
        with pytest.raises(DatabaseError):
            MediaDatabase(str(path), client_id="fail")

    survivor = MediaDatabase.__new__(MediaDatabase)  # avoid re-running the chain
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        assert conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()["version"] == 8
        # No half-applied index survived the rollback...
        assert not set(V9_INDEXES) & set(_indexes_on_media(conn))
        # ...and the database still answers on the old plan.
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM Media WHERE deleted = 0 AND is_trash = 0"
        ).fetchone()["n"] == 8
    finally:
        conn.close()
        del survivor


# ---------------------------------------------------------------------------
# The load-bearing set: the planner picks them with NO sqlite_stat1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(LIST_QUERIES))
def test_list_query_plan_uses_a_v9_index_without_analyze(fresh_db, label):
    sql, params, expected = LIST_QUERIES[label]
    conn = fresh_db.get_connection()
    _seed(conn)
    plan = _plan(conn, sql, params)
    if expected is not None:
        assert expected in plan, f"{label}: {plan}"
    else:
        assert any(name in plan for name in V9_INDEXES), f"{label}: {plan}"
    # The pre-v9 plan for every one of these was
    # "SEARCH ... USING INDEX idx_media_deleted (deleted=?)".
    assert "idx_media_deleted" not in plan, f"{label}: {plan}"


@pytest.mark.parametrize("label", NO_TEMP_SORT)
def test_list_query_no_longer_sorts_the_whole_library(fresh_db, label):
    sql, params, _expected = LIST_QUERIES[label]
    conn = fresh_db.get_connection()
    _seed(conn)
    plan = _plan(conn, sql, params)
    assert "TEMP B-TREE" not in plan, f"{label}: {plan}"


def test_the_textbook_index_shape_is_never_chosen_without_stats(tmp_path):
    """The negative control, and the whole reason v9 leads with `deleted`.

    ``(last_modified DESC, id DESC) WHERE deleted = 0 AND is_trash = 0`` is
    a perfect index for the Library page query. With no ``sqlite_stat1`` the
    planner ignores it and keeps its one-column ``idx_media_deleted`` search
    plus the temp B-tree -- which is what makes "the index exists" a
    worthless assertion and this whole file necessary.
    """
    db = MediaDatabase(str(tmp_path / "control.db"), client_id="control")
    try:
        conn = db.get_connection()
        _seed(conn)
        for name in V9_INDEXES:
            conn.execute(f"DROP INDEX {name}")
        conn.execute(
            "CREATE INDEX idx_media_textbook ON Media(last_modified DESC, id DESC) "
            "WHERE deleted = 0 AND is_trash = 0"
        )
        conn.commit()
        sql, params, _ = LIST_QUERIES["list_library_media_page rows"]
        plan = _plan(conn, sql, params)
        assert "idx_media_textbook" not in plan, plan
        assert "TEMP B-TREE" in plan, plan
    finally:
        db.close_connection()


def test_list_results_are_unchanged_by_the_indexes(fresh_db):
    """The indexes must not move a single row or reorder a single page."""
    conn = fresh_db.get_connection()
    _seed(conn)
    sql, params, _ = LIST_QUERIES["list_library_media_page rows"]
    with_indexes = [tuple(r) for r in conn.execute(sql, params)]
    deep = (PREVIEW, 25, 300)
    with_indexes_deep = [tuple(r) for r in conn.execute(sql, deep)]

    for name in V9_INDEXES:
        conn.execute(f"DROP INDEX {name}")
    conn.commit()

    assert [tuple(r) for r in conn.execute(sql, params)] == with_indexes
    assert [tuple(r) for r in conn.execute(sql, deep)] == with_indexes_deep
    assert len(with_indexes) == 25


def test_trashed_and_soft_deleted_rows_are_outside_the_partial_indexes(fresh_db):
    """A partial index must hold exactly the rows the readers can see."""
    conn = fresh_db.get_connection()
    _seed(conn, live=50, trashed=7, deleted=9)
    live = conn.execute(
        "SELECT COUNT(*) AS n FROM Media WHERE deleted = 0 AND is_trash = 0"
    ).fetchone()["n"]
    assert live == 50
    rows, total = fresh_db.search_media_db(search_query=None, results_per_page=100)
    assert total == 50 and len(rows) == 50
    # The trash and deleted views still work -- they never used these indexes.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM Media WHERE is_trash = 1 AND deleted = 0"
    ).fetchone()["n"] == 7


def test_empty_library_still_plans_through_the_indexes(fresh_db):
    """First run: no media at all. No crash, no scan, honest empty page."""
    conn = fresh_db.get_connection()
    sql, params, _ = LIST_QUERIES["list_library_media_page rows"]
    assert conn.execute(sql, params).fetchall() == []
    assert RECENT in _plan(conn, sql, params)


# ---------------------------------------------------------------------------
# search_media_db: the keyword subqueries reach Keywords by its unique index
# ---------------------------------------------------------------------------


MUST_HAVE_SQL = (
    f"SELECT DISTINCT {BROAD} FROM Media m WHERE m.deleted = 0 AND m.is_trash = 0 "
    f"AND ((SELECT COUNT(DISTINCT k_mh.id) FROM MediaKeywords mk_mh "
    f"JOIN Keywords k_mh ON mk_mh.keyword_id = k_mh.id "
    f"WHERE mk_mh.media_id = m.id AND k_mh.deleted = 0 "
    f"AND k_mh.keyword IN (?)) = ?) "
    f"ORDER BY m.last_modified DESC, m.id DESC LIMIT ? OFFSET ?"
)


def test_must_have_keywords_reaches_keywords_by_its_unique_index(fresh_db):
    conn = fresh_db.get_connection()
    _seed(conn)
    plan = _plan(conn, MUST_HAVE_SQL, ("kw1", 1, 20, 0))
    # The pre-fix plan walked EVERY live keyword per candidate media row:
    # "SEARCH k_mh USING INDEX idx_keywords_deleted (deleted=?)".
    assert "sqlite_autoindex_Keywords_1 (keyword=?)" in plan, plan
    assert "idx_keywords_deleted" not in plan, plan


def test_must_not_have_keywords_reaches_keywords_by_its_unique_index(fresh_db):
    conn = fresh_db.get_connection()
    _seed(conn)
    sql = (
        f"SELECT DISTINCT {BROAD} FROM Media m WHERE m.deleted = 0 "
        f"AND m.is_trash = 0 AND (NOT EXISTS (SELECT 1 FROM MediaKeywords mk_mnh "
        f"JOIN Keywords k_mnh ON mk_mnh.keyword_id = k_mnh.id "
        f"WHERE mk_mnh.media_id = m.id AND k_mnh.deleted = 0 "
        f"AND k_mnh.keyword IN (?))) "
        f"ORDER BY m.last_modified DESC, m.id DESC LIMIT ? OFFSET ?"
    )
    plan = _plan(conn, sql, ("kw1", 20, 0))
    assert "sqlite_autoindex_Keywords_1 (keyword=?)" in plan, plan


def test_the_production_sql_no_longer_wraps_the_keyword_column(fresh_db):
    """The rewrite is in the BUILDER, not only in this file's literals.

    Comment lines are stripped first: the fix carries a long comment that
    quotes the old spelling, and a naive substring check would fail on the
    explanation of the very change it is pinning.
    """
    import inspect

    code = "\n".join(
        line
        for line in inspect.getsource(MediaDatabase.search_media_db).splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "LOWER(k_mh.keyword)" not in code
    assert "LOWER(k_mnh.keyword)" not in code
    assert "k_mh.keyword IN (" in code
    assert "k_mnh.keyword IN (" in code


@pytest.mark.parametrize(
    "stored, typed",
    [
        ("dragon", "dragon"),
        ("Dragon", "dragon"),
        ("DRAGON", "dragon"),
        ("DrAgOn", "dragon"),
        ("café", "café"),
        ("CAFÉ", "café"),
        ("ünïcodé", "ünïcodé"),
        ("a b", "a b"),
        ("a%b", "a%b"),
        ("a_b", "a_b"),
        ("1a", "1a"),
        ("straße", "straße"),
        ("İstanbul", "i̇stanbul"),
    ],
)
def test_bare_keyword_match_equals_the_lower_spelling_for_every_bindable_value(
    stored, typed
):
    """`Keywords.keyword` is UNIQUE COLLATE NOCASE and the caller lowercases.

    The two spellings can only diverge for a bound value containing
    uppercase, which ``search_media_db`` cannot produce -- it does
    ``k.strip().lower()`` first. `typed` here is exactly what that
    expression yields.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE Keywords (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "keyword TEXT NOT NULL UNIQUE COLLATE NOCASE, deleted BOOLEAN NOT NULL "
        "DEFAULT 0)"
    )
    conn.execute("INSERT INTO Keywords (keyword) VALUES (?)", (stored,))
    conn.commit()
    assert typed == typed.strip().lower(), "the fixture must bind what the caller binds"
    lowered = conn.execute(
        "SELECT id FROM Keywords WHERE deleted = 0 AND LOWER(keyword) IN (?)", (typed,)
    ).fetchall()
    bare = conn.execute(
        "SELECT id FROM Keywords WHERE deleted = 0 AND keyword IN (?)", (typed,)
    ).fetchall()
    assert [tuple(r) for r in bare] == [tuple(r) for r in lowered]
    conn.close()


def test_must_have_keywords_returns_the_same_media_as_before(fresh_db):
    """End-to-end through the real builder, against a case-mixed corpus."""
    conn = fresh_db.get_connection()
    _seed(conn, live=30, trashed=2, deleted=2, keywords=4)
    conn.execute(
        "INSERT INTO Keywords (id, keyword, uuid, last_modified, version, "
        "client_id, deleted) VALUES (99, 'MixedCase', 'kw-uuid-99', "
        "'2024-05-01T00:00:00Z', 1, 'test', 0)"
    )
    conn.executemany(
        "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, 99)",
        [(1,), (2,), (3,)],
    )
    conn.commit()
    for typed in ("mixedcase", "MixedCase", "MIXEDCASE"):
        rows, total = fresh_db.search_media_db(
            search_query=None, must_have_keywords=[typed], results_per_page=50
        )
        assert total == 3, typed
        assert sorted(r["id"] for r in rows) == [1, 2, 3], typed


def test_must_not_have_keywords_still_excludes_case_insensitively(fresh_db):
    conn = fresh_db.get_connection()
    _seed(conn, live=10, trashed=0, deleted=0, keywords=3)
    conn.execute(
        "INSERT INTO Keywords (id, keyword, uuid, last_modified, version, "
        "client_id, deleted) VALUES (99, 'Excluded', 'kw-uuid-99', "
        "'2024-05-01T00:00:00Z', 1, 'test', 0)"
    )
    conn.executemany(
        "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, 99)",
        [(4,), (5,)],
    )
    conn.commit()
    rows, total = fresh_db.search_media_db(
        search_query=None, must_not_have_keywords=["EXCLUDED"], results_per_page=50
    )
    assert total == 8
    assert 4 not in {r["id"] for r in rows} and 5 not in {r["id"] for r in rows}


# ---------------------------------------------------------------------------
# search_media_db: the FTS COUNT join order
# ---------------------------------------------------------------------------


def _count_sql_for(db: MediaDatabase, _unused: list, **kwargs) -> str:
    """Run ``search_media_db`` and return the COUNT statement it issued.

    Uses SQLite's own trace callback rather than a wrapper, so what is
    captured is the text that genuinely reached the engine -- there is no
    seam left for a paraphrase to slip through.
    """
    conn = db.get_connection()
    captured: list[str] = []
    conn.set_trace_callback(captured.append)
    try:
        db.search_media_db(**kwargs)
    finally:
        conn.set_trace_callback(None)
    matches = [s for s in captured if "COUNT(DISTINCT m.id)" in s]
    assert matches, f"no COUNT statement was traced; saw {captured}"
    return matches[0]


def test_fts_count_pins_media_fts_as_the_outer_loop(fresh_db):
    conn = fresh_db.get_connection()
    _seed(conn)
    count_sql = _count_sql_for(
        fresh_db, [], search_query="dragon", results_per_page=10
    )
    assert "FROM media_fts fts CROSS JOIN Media m" in count_sql, count_sql
    # The traced text already carries its bound values inline, so it plans
    # with no parameters -- and it is the engine's own copy, not a paraphrase.
    plan = _plan(conn, count_sql)
    # Pre-fix: "SEARCH m USING INDEX idx_media_deleted | SCAN fts ..." --
    # one FTS probe per live media row, measured 276 ms at 20k media.
    assert plan.index("SCAN fts") < plan.index("SEARCH m"), plan


def test_fts_count_keeps_media_first_when_an_id_allowlist_is_given(fresh_db):
    """The measured exception: a small allowlist makes Media the cheap side.

    Pinning fts first there costs 0.10 -> 1.92 ms, so the pin is withheld.
    """
    conn = fresh_db.get_connection()
    _seed(conn)
    count_sql = _count_sql_for(
        fresh_db,
        [],
        search_query="dragon",
        media_ids_filter=[1, 2, 3, 4, 5],
        results_per_page=10,
    )
    assert "CROSS JOIN" not in count_sql, count_sql
    assert "FROM Media m JOIN media_fts fts" in count_sql, count_sql


def test_fts_search_returns_the_same_rows_and_total_either_way(fresh_db):
    conn = fresh_db.get_connection()
    _seed(conn)
    rows, total = fresh_db.search_media_db(
        search_query="dragon", results_per_page=25
    )
    reference = conn.execute(
        "SELECT COUNT(DISTINCT m.id) FROM Media m "
        "JOIN media_fts fts ON fts.rowid = m.id "
        "WHERE m.deleted = 0 AND m.is_trash = 0 AND fts.media_fts MATCH ? "
        "AND ((m.title LIKE ? COLLATE NOCASE OR m.content LIKE ? COLLATE NOCASE))",
        ('"dragon"', "%dragon%", "%dragon%"),
    ).fetchone()[0]
    assert total == reference > 0
    assert len(rows) == 25

    scoped, scoped_total = fresh_db.search_media_db(
        search_query="dragon", media_ids_filter=[1, 2, 3], results_per_page=25
    )
    assert scoped_total == len(scoped) == 3


def test_non_fts_search_is_unchanged_by_the_count_rewrite(fresh_db):
    conn = fresh_db.get_connection()
    _seed(conn)
    count_sql = _count_sql_for(fresh_db, [], search_query=None, results_per_page=10)
    assert count_sql.strip().startswith("SELECT COUNT(DISTINCT m.id) FROM Media m")
    assert "media_fts" not in count_sql
