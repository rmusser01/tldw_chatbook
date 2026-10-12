"""Native historical author catalogs, populated upgrades and refusal boundaries."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB, DreamsSchemaError

HISTORY = json.loads((Path(__file__).parent / "fixtures/schema_history.json").read_text())
LAYOUTS = HISTORY["layouts"]


def historical_store(path, layout, *, populate=True):
    """Build the measured historical catalog, never a rewound current schema."""
    with closing(sqlite3.connect(path)) as conn:
        # Tables precede indexes; SQLite creates its own sqlite_sequence.
        statements = [HISTORY["statements"][index] for index in layout["statements"]]
        for sql in sorted(statements, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.executemany("INSERT INTO schema_version VALUES (?)",
                         [(version,) for version in layout["version_rows"]])
        if not populate:
            conn.commit()
            return
        conn.executemany("""INSERT INTO dream_interest_profile
            (id,facet,text,weight,searchable,source,query_angle,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)""", [
                (31,"topic","native topic",2.5,1,"seed","seed angle","old-created","old-updated"),
                (32,"goal","private goal",3.5,0,"user","private angle","old-created","old-updated")])
        conn.execute("""INSERT INTO dreams_collections
            (id,local_date,status,profile_digest,trigger,created_at)
            VALUES (41,'2020-01-02','complete','digest','manual','old-created')""")
        conn.execute("""INSERT INTO dream_stories
            (id,collection_id,title,url,snippet,body,status,source,kind,matched_topics,query,created_at)
            VALUES (51,41,'story','https://example.test/story','snippet','body',
                'complete','web','content','["native topic"]','exact query','old-created')""")
        conn.execute("INSERT INTO dream_feedback VALUES (61,51,'more','old-created')")
        conn.execute("INSERT INTO dream_seen_items VALUES ('https://example.test/story','digest','first','last')")
        conn.execute("INSERT INTO dream_daily_usage VALUES ('2020-01-02',3,4)")
        if layout["version"] >= 2:
            conn.execute("""INSERT INTO dream_tracked_items
                (id,origin_story_id,mechanism,intent,subscription_id,cadence_seconds,created_at,updated_at)
                VALUES (71,51,'page','topic',81,3600,'old-created','old-updated')""")
            conn.execute("""INSERT INTO dream_track_runs
                (id,tracked_item_id,status,digest_hash,created_at)
                VALUES (91,71,'baseline','track-digest','old-created')""")
        if layout["version"] >= 4:
            conn.execute("UPDATE dream_stories SET goal='original generating goal'")
        if layout["version"] >= 5:
            conn.execute("UPDATE dream_tracked_items SET suspension='original suspension'")
        conn.commit()


def snapshot(conn):
    return {name: ([row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')],
                   conn.execute(f'SELECT * FROM "{name}"').fetchall())
            for (name,) in conn.execute("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")}


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda row: f"v{row['version']}-{row['source_commit'][:8]}-{row['statements'][-2]}")
def test_populated_historical_catalog_upgrades_and_reopens(tmp_path, layout):
    path = tmp_path / "dreams.sqlite"
    historical_store(path, layout)
    with closing(sqlite3.connect(path)) as conn:
        original = snapshot(conn)
    db = DreamsDB(path)
    try:
        with db.connection() as conn:
            for table, (columns, rows) in original.items():
                if table == "schema_version":
                    assert {row[0] for row in rows} <= {row[0] for row in conn.execute("SELECT version FROM schema_version")}
                    continue
                projection = ",".join(f'"{column}"' for column in columns)
                assert [tuple(row) for row in conn.execute(f'SELECT {projection} FROM "{table}"')] == rows
            assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 6
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
            after = "\n".join(conn.iterdump())
    finally:
        db.close_all()
    db = DreamsDB(path)
    try:
        with db.connection() as conn:
            assert "\n".join(conn.iterdump()) == after
    finally:
        db.close_all()


@pytest.mark.parametrize("mutation", [
    "DROP TABLE schema_version", "DELETE FROM schema_version",
    "INSERT INTO schema_version VALUES (99)", "INSERT INTO schema_version VALUES (0)",
    "DROP TABLE dream_feedback", "ALTER TABLE dream_stories ADD COLUMN foreign_field TEXT",
])
def test_unknown_or_hybrid_store_refused_unchanged(tmp_path, mutation):
    path = tmp_path / "dreams.sqlite"
    historical_store(path, LAYOUTS[-1])
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(mutation)
        conn.commit()
        original = "\n".join(conn.iterdump())
    with pytest.raises(DreamsSchemaError):
        DreamsDB(path)
    with closing(sqlite3.connect(path)) as conn:
        assert "\n".join(conn.iterdump()) == original


def test_failed_upgrade_rolls_back_and_reopens(tmp_path, monkeypatch):
    path = tmp_path / "dreams.sqlite"
    historical_store(path, LAYOUTS[0])
    with closing(sqlite3.connect(path)) as conn:
        original = "\n".join(conn.iterdump())
    migrate = DreamsDB._migrate_story_goal_v4
    def fail(conn):
        migrate(conn)
        raise RuntimeError("injected after migration write")
    with monkeypatch.context() as patch:
        patch.setattr(DreamsDB, "_migrate_story_goal_v4", staticmethod(fail))
        with pytest.raises(RuntimeError, match="injected"):
            DreamsDB(path)
    with closing(sqlite3.connect(path)) as conn:
        assert "\n".join(conn.iterdump()) == original
    db = DreamsDB(path)
    db.close_all()


def replace_with_historical_fixture(db, version):
    """Preserve original test data in a genuine historical temporary schema."""
    with db.connection() as conn:
        original = snapshot(conn)
    db.close_all()
    replacement = db.db_path.with_suffix(".historical")
    historical_store(replacement, next(row for row in LAYOUTS if row["version"] == version), populate=False)
    with closing(sqlite3.connect(replacement)) as conn:
        for table, (old_columns, rows) in original.items():
            if table == "schema_version":
                continue
            columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            if not columns:
                continue
            selected = [column for column in columns if column in old_columns]
            projection = ",".join(f'"{column}"' for column in selected)
            placeholders = ",".join("?" for _ in selected)
            conn.executemany(f'INSERT INTO "{table}" ({projection}) VALUES ({placeholders})',
                             [tuple(row[old_columns.index(column)] for column in selected) for row in rows])
        conn.commit()
    replacement.replace(db.db_path)


def test_refused_constructor_closes_native_handle_before_return(tmp_path, monkeypatch):
    from tldw_chatbook.DB.base_db import BaseDB

    path = tmp_path / "future.sqlite"
    historical_store(path, LAYOUTS[-1])
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO schema_version VALUES (99)")
        conn.commit()
    original = path.read_bytes()
    opened = []
    acquire = BaseDB._get_connection
    def capture(self, *args, **kwargs):
        conn = acquire(self, *args, **kwargs)
        opened.append(conn)
        return conn
    monkeypatch.setattr(BaseDB, "_get_connection", capture)
    with pytest.raises(DreamsSchemaError, match="schema_too_new"):
        DreamsDB(path)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    assert path.read_bytes() == original
    assert not path.with_name(path.name + "-wal").exists()
