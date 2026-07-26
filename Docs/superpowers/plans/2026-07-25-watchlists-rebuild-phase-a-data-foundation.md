# Watchlists Rebuild — Phase A: Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the storage and service layer the Watchlists screen rebuild needs — the watchlist bundle entity, durable item body text, full-text search, and a single item-persist path — with no UI changes.

**Architecture:** Everything extends `SubscriptionsDB`'s existing idempotent `_ensure_watchlists_schema` helper rather than introducing migration machinery. A new `WatchlistBundleService` owns the watchlist entity and membership. The two divergent item INSERT paths collapse into one shared function. No UI file is touched in this phase.

**Tech Stack:** Python ≥3.11, SQLite with FTS5, pytest. No new dependencies.

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`. Every requirement below is traceable to it.
- All schema changes go through `SubscriptionsDB._ensure_watchlists_schema` and must be **idempotent** — running twice is a no-op.
- `PRAGMA foreign_keys = ON` is already set (`Subscriptions_DB.py:82`). Cascades are live; do not disable.
- `SubscriptionsDB` uses `threading.local()` connections (`:75`, `:368-370`). Never share a connection across threads.
- Use `db.transaction()` (`:373`) for writes; it commits on success and rolls back on exception.
- `subscriptions.tags` is **comma-joined**, not JSON (`:422`). `watchlists.tags` matches.
- `subscription_items.status` CHECK allows only `new`, `reviewed`, `ingested`, `ignored`, `error` (`:156`). Do not add values. "read" means `reviewed`; flag is the separate `is_flagged` column.
- Tests run from the venv: `source .venv/bin/activate && pytest`. The `timeout` command is unavailable in this environment.
- Parameterized queries only. Never interpolate values into SQL.

## Phase Map

This plan is Phase A of five. Later phases get their own plans, written against merged code rather than speculation:

| Phase | Scope | Depends on |
|---|---|---|
| **A (this plan)** | Schema, unified persist, FTS, bundle service, counts | — |
| B | `watchlists_workbench.py` container + five-region collapse (`region_layout.py`) + shell rewrite | A |
| C | Tree, feeds pane, items pane, Inspector breadcrumb stack | B |
| D | Content pane: article + change renderers, `content_render.py`, escaping | C |
| E | Sources / Runs / Rules / Artifacts tabs | C |

Phase A ships as its own PR and is independently testable — it has no UI surface.

## File Structure

| Path | Responsibility | Task |
|---|---|---|
| `tldw_chatbook/DB/Subscriptions_DB.py` | Schema additions, FTS, backfill, count query | 1, 3, 7 |
| `tldw_chatbook/Subscriptions/local_watchlists_service.py` | `batch_id` on lazy run table; delegate persist | 2, 4 |
| `tldw_chatbook/Subscriptions/item_persist.py` | **New.** Single item-persist function for both callers | 4 |
| `tldw_chatbook/Subscriptions/watchlist_bundle_service.py` | **New.** Watchlist CRUD, membership, folder migration | 5, 6 |
| `Tests/DB/test_subscriptions_db_watchlists.py` | **New.** Schema, FTS, backfill, counts | 1, 3, 7 |
| `Tests/Subscriptions/test_item_persist.py` | **New.** Full-column-set persistence | 4 |
| `Tests/Subscriptions/test_watchlist_bundle_service.py` | **New.** CRUD, membership, collisions, migration | 5, 6 |

---

### Task 1: Watchlist tables and item columns

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` — inside `_ensure_watchlists_schema`, after the existing index creation at `:361-362`
- Test: `Tests/DB/test_subscriptions_db_watchlists.py`

**Interfaces:**
- Consumes: nothing.
- Produces: tables `watchlists`, `watchlist_sources`, `watchlist_migration_state`; columns `subscription_items.content`, `.content_format`, `.content_kind`, `.is_flagged`. Tasks 3-7 all depend on these.

- [ ] **Step 1: Write the failing test**

Create `Tests/DB/test_subscriptions_db_watchlists.py`:

```python
import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


def _columns(db, table):
    cursor = db.conn.cursor()
    return {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}


def _tables(db):
    cursor = db.conn.cursor()
    return {
        row[0]
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_watchlist_tables_created(db):
    tables = _tables(db)
    assert "watchlists" in tables
    assert "watchlist_sources" in tables
    assert "watchlist_migration_state" in tables


def test_item_content_columns_created(db):
    cols = _columns(db, "subscription_items")
    assert "content" in cols
    assert "content_format" in cols
    assert "content_kind" in cols
    assert "is_flagged" in cols


def test_membership_cascades_on_source_delete(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    with db.transaction() as conn:
        conn.execute("INSERT INTO watchlists (name) VALUES ('Morning')")
        watchlist_id = conn.execute("SELECT id FROM watchlists").fetchone()[0]
        conn.execute(
            "INSERT INTO watchlist_sources (watchlist_id, subscription_id) VALUES (?, ?)",
            (watchlist_id, source_id),
        )

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (source_id,))

    remaining = db.conn.execute("SELECT COUNT(*) FROM watchlist_sources").fetchone()[0]
    assert remaining == 0


def test_membership_cascades_on_watchlist_delete(db):
    source_id = db.add_subscription(name="HN", type="rss", source="https://b.example/f")
    with db.transaction() as conn:
        conn.execute("INSERT INTO watchlists (name) VALUES ('Security')")
        watchlist_id = conn.execute("SELECT id FROM watchlists").fetchone()[0]
        conn.execute(
            "INSERT INTO watchlist_sources (watchlist_id, subscription_id) VALUES (?, ?)",
            (watchlist_id, source_id),
        )
        conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))

    remaining = db.conn.execute("SELECT COUNT(*) FROM watchlist_sources").fetchone()[0]
    assert remaining == 0
    # The source itself survives — only membership is removed.
    assert db.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 1


def test_schema_migration_is_idempotent(db):
    db._ensure_watchlists_schema()
    db._ensure_watchlists_schema()
    assert "watchlists" in _tables(db)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db_watchlists.py -v`
Expected: FAIL — `test_watchlist_tables_created` asserts `"watchlists" in tables` and the table does not exist.

- [ ] **Step 3: Write minimal implementation**

In `tldw_chatbook/DB/Subscriptions_DB.py`, inside `_ensure_watchlists_schema`, **replace** the two index lines at `:361-362` with the block below (keeping those two `CREATE INDEX` statements at the top of it):

```python
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscription_items_run_id ON subscription_items(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscription_items_queued ON subscription_items(queued_for_briefing, status)")

        # Reader/body columns. `content` holds the renderable body: article text
        # for feed items, diff text for site changes. `url_snapshots` remains the
        # authority for full-page and previous-snapshot views.
        if "content" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN content TEXT")
        if "content_format" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN content_format TEXT")
        if "content_kind" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN content_kind TEXT")
        # Flag is a separate boolean, not a status: the status CHECK has no
        # 'flagged' value, and an item can be flagged *and* reviewed at once.
        if "is_flagged" not in items_cols:
            cursor.execute("ALTER TABLE subscription_items ADD COLUMN is_flagged BOOLEAN DEFAULT 0")

        # Watchlist bundle entity. `name` is intentionally not UNIQUE — uniqueness
        # is enforced case-insensitively in WatchlistBundleService with
        # auto-suffixing, because a SQL constraint would raise mid-migration.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                is_active BOOLEAN DEFAULT 1,
                sort_order INTEGER DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_sources (
                watchlist_id    INTEGER NOT NULL REFERENCES watchlists(id)     ON DELETE CASCADE,
                subscription_id INTEGER NOT NULL REFERENCES subscriptions(id)  ON DELETE CASCADE,
                added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (watchlist_id, subscription_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_watchlist_sources_subscription "
            "ON watchlist_sources(subscription_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_subscription_items_flagged "
            "ON subscription_items(is_flagged, status)"
        )
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_watchlists_timestamp
            AFTER UPDATE ON watchlists
            BEGIN
                UPDATE watchlists SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """)
        # Per-migration markers. schema_version cannot be reused: it is a single
        # INTEGER PRIMARY KEY column with no room for keys.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_migration_state (
                key TEXT PRIMARY KEY,
                applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
```

Delete the now-duplicated trailing `conn.commit()` that followed the original index lines.

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db_watchlists.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Run the existing DB suite for regressions**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db.py -v`
Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py Tests/DB/test_subscriptions_db_watchlists.py
git commit -m "feat(watchlists): add watchlist tables and item content columns"
```

---

### Task 2: batch_id on the lazily-created run table

**Files:**
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py:949-968` (`_ensure_run_schema`)
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` — `_ensure_watchlists_schema`
- Test: `Tests/DB/test_subscriptions_db_watchlists.py`

**Interfaces:**
- Consumes: Task 1's `_ensure_watchlists_schema` block.
- Produces: `local_watchlist_runs.batch_id TEXT`, present whether the table is created fresh or already exists.

**Why this is delicate:** `local_watchlist_runs` is created **lazily** by `LocalWatchlistsService._ensure_run_schema`, not by `SubscriptionsDB._initialize_schema`. An unconditional `ALTER TABLE local_watchlist_runs` inside `_ensure_watchlists_schema` fails on a fresh database because the table does not exist yet. The column must therefore be added in **both** places: the lazy `CREATE TABLE` for new databases, and a **conditional** `ALTER` for databases where the table already exists.

- [ ] **Step 1: Write the failing test**

Append to `Tests/DB/test_subscriptions_db_watchlists.py`:

```python
def test_batch_id_migration_skips_missing_run_table(db):
    # Fresh DB: local_watchlist_runs does not exist yet. The migration must not raise.
    assert "local_watchlist_runs" not in _tables(db)
    db._ensure_watchlists_schema()


def test_batch_id_added_to_existing_run_table(db):
    # Simulate a DB created before batch_id existed.
    with db.transaction() as conn:
        conn.execute("""
            CREATE TABLE local_watchlist_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                job_id INTEGER,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                stats_json TEXT,
                error_msg TEXT,
                log_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

    db._ensure_watchlists_schema()
    assert "batch_id" in _columns(db, "local_watchlist_runs")


def test_batch_id_present_on_lazily_created_run_table(db):
    from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService

    LocalWatchlistsService._ensure_run_schema(db)
    assert "batch_id" in _columns(db, "local_watchlist_runs")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db_watchlists.py -k batch_id -v`
Expected: FAIL — `test_batch_id_added_to_existing_run_table` and `test_batch_id_present_on_lazily_created_run_table` fail because `batch_id` does not exist.

- [ ] **Step 3: Add the column to the lazy CREATE TABLE**

In `tldw_chatbook/Subscriptions/local_watchlists_service.py`, in `_ensure_run_schema`, add `batch_id` after `job_id`:

```python
                CREATE TABLE IF NOT EXISTS local_watchlist_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    job_id INTEGER,
                    batch_id TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    stats_json TEXT,
                    error_msg TEXT,
                    log_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES subscriptions(id) ON DELETE CASCADE
                )
```

- [ ] **Step 4: Add the conditional ALTER**

In `tldw_chatbook/DB/Subscriptions_DB.py`, inside `_ensure_watchlists_schema`, immediately before the final `conn.commit()` added in Task 1:

```python
        # local_watchlist_runs is created lazily by LocalWatchlistsService, so it
        # may not exist yet. ALTER only when it does; new databases get batch_id
        # from that service's CREATE TABLE instead.
        run_table_exists = cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='local_watchlist_runs'"
        ).fetchone()
        if run_table_exists:
            run_cols = {row[1] for row in cursor.execute("PRAGMA table_info(local_watchlist_runs)")}
            if "batch_id" not in run_cols:
                cursor.execute("ALTER TABLE local_watchlist_runs ADD COLUMN batch_id TEXT")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_local_watchlist_runs_batch "
                "ON local_watchlist_runs(batch_id)"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db_watchlists.py -v && pytest Tests/Watchlists -v`
Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Subscriptions/local_watchlists_service.py Tests/DB/test_subscriptions_db_watchlists.py
git commit -m "feat(watchlists): add batch_id to run table with conditional migration"
```

---

### Task 3: FTS5 index over items

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` — `_ensure_watchlists_schema`, plus a new `backfill_items_fts` method
- Test: `Tests/DB/test_subscriptions_db_watchlists.py`

**Interfaces:**
- Consumes: Task 1's `content` column. The FTS triggers reference `new.content`, so the column must exist before the triggers are created — keep this block **after** Task 1's.
- Produces: `subscription_items_fts` virtual table; `SubscriptionsDB.backfill_items_fts(chunk_size: int = 500) -> int` returning the number of rows indexed in this call, `0` when complete.

**Trigger caution:** an `update_subscription_items_timestamp` trigger already exists (`Subscriptions_DB.py:274-278`) and issues its own `UPDATE subscription_items`. SQLite's `recursive_triggers` pragma is off by default, so that nested update does **not** re-fire the FTS trigger. Do not enable `recursive_triggers`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/DB/test_subscriptions_db_watchlists.py`:

```python
def _insert_item(db, subscription_id, url, title, content):
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscription_items "
            "(subscription_id, url, title, content, content_kind, content_format) "
            "VALUES (?, ?, ?, ?, 'article', 'text')",
            (subscription_id, url, title, content),
        )


def test_fts_table_created(db):
    assert "subscription_items_fts" in _tables(db)


def test_fts_indexes_inserted_items(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/1", "RAG Evaluation", "retrieval quality rubric")

    rows = db.conn.execute(
        "SELECT rowid FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("rubric",),
    ).fetchall()
    assert len(rows) == 1


def test_fts_follows_updates_and_deletes(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/1", "First", "alpha content")

    with db.transaction() as conn:
        conn.execute("UPDATE subscription_items SET content = 'beta content' WHERE url = ?",
                     ("https://a.example/1",))

    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("alpha",),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("beta",),
    ).fetchone()[0] == 1

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items WHERE url = ?", ("https://a.example/1",))

    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("beta",),
    ).fetchone()[0] == 0


def test_backfill_is_chunked_and_resumable(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER subscription_items_fts_ai")
        for index in range(7):
            conn.execute(
                "INSERT INTO subscription_items (subscription_id, url, title, content) "
                "VALUES (?, ?, ?, ?)",
                (source_id, f"https://a.example/{index}", f"Item {index}", "searchable body"),
            )

    assert db.conn.execute("SELECT COUNT(*) FROM subscription_items_fts").fetchone()[0] == 0

    first = db.backfill_items_fts(chunk_size=3)
    assert first == 3
    total = first
    while True:
        indexed = db.backfill_items_fts(chunk_size=3)
        if indexed == 0:
            break
        total += indexed
    assert total == 7

    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("searchable",),
    ).fetchone()[0] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db_watchlists.py -k "fts or backfill" -v`

The `-k` expression must be quoted — unquoted, the shell parses `or` as a separate argument.

Expected: FAIL — `test_fts_table_created` fails, `subscription_items_fts` does not exist.

- [ ] **Step 3: Create the FTS table and triggers**

In `tldw_chatbook/DB/Subscriptions_DB.py`, inside `_ensure_watchlists_schema`, after Task 2's block and before the final `conn.commit()`:

```python
        # External-content FTS over items, matching the pattern used by
        # character_cards_fts / conversations_fts / media_fts. Triggers rather
        # than explicit index writes, so every INSERT path stays indexed.
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS subscription_items_fts USING fts5(
                title,
                content,
                author,
                content='subscription_items',
                content_rowid='id'
            )
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS subscription_items_fts_ai
            AFTER INSERT ON subscription_items BEGIN
                INSERT INTO subscription_items_fts(rowid, title, content, author)
                VALUES (new.id, new.title, new.content, new.author);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS subscription_items_fts_ad
            AFTER DELETE ON subscription_items BEGIN
                INSERT INTO subscription_items_fts(subscription_items_fts, rowid, title, content, author)
                VALUES ('delete', old.id, old.title, old.content, old.author);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS subscription_items_fts_au
            AFTER UPDATE ON subscription_items BEGIN
                INSERT INTO subscription_items_fts(subscription_items_fts, rowid, title, content, author)
                VALUES ('delete', old.id, old.title, old.content, old.author);
                INSERT INTO subscription_items_fts(rowid, title, content, author)
                VALUES (new.id, new.title, new.content, new.author);
            END
        """)
```

- [ ] **Step 4: Add the chunked backfill method**

Add as a public method on `SubscriptionsDB`, directly after `_ensure_watchlists_schema`:

```python
    def backfill_items_fts(self, chunk_size: int = 500) -> int:
        """Index one chunk of items that are missing from the FTS table.

        Runs in a background worker, never inline in migration — a synchronous
        backfill of a large subscription_items table would block app boot.
        Resumes by rowid, so an interrupted backfill continues rather than
        restarting.

        Args:
            chunk_size: Maximum rows to index in this call.

        Returns:
            Number of rows indexed. ``0`` means the backfill is complete.
        """
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, title, content, author
                FROM subscription_items
                WHERE id NOT IN (SELECT rowid FROM subscription_items_fts)
                ORDER BY id
                LIMIT ?
                """,
                (chunk_size,),
            ).fetchall()
            if not rows:
                return 0
            conn.executemany(
                "INSERT INTO subscription_items_fts(rowid, title, content, author) "
                "VALUES (?, ?, ?, ?)",
                [(row[0], row[1], row[2], row[3]) for row in rows],
            )
            return len(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db_watchlists.py -v`
Expected: PASS, all tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py Tests/DB/test_subscriptions_db_watchlists.py
git commit -m "feat(watchlists): add FTS5 index over items with chunked backfill"
```

---

### Task 4: Unified item persistence

**Files:**
- Create: `tldw_chatbook/Subscriptions/item_persist.py`
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py:1296-1340`
- Test: `Tests/Subscriptions/test_item_persist.py`

**Interfaces:**
- Consumes: Task 1's `content`, `content_format`, `content_kind` columns.
- Produces: `persist_subscription_item(conn, subscription_id: int, item: Mapping[str, Any], run_id: int | None, now: str) -> None`.

**The bug being fixed:** two INSERT paths write disjoint column sets. `Subscriptions_DB.py:1322` writes `canonical_url`, `previous_hash`, `change_percentage`, `diff_summary`, `change_type` but drops `status`/`run_id`/`alert_matches`. `local_watchlists_service.py:1301` does the reverse. Neither writes body text, even though `local_watchlists_service.py:878` normalizes a `content` value before discarding it.

- [ ] **Step 1: Write the failing test**

Create `Tests/Subscriptions/test_item_persist.py`:

```python
import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


@pytest.fixture
def source_id(db):
    return db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")


def test_persists_full_column_set(db, source_id):
    item = {
        "url": "https://a.example/1",
        "title": "RAG Evaluation",
        "content": "retrieval quality rubric",
        "content_kind": "article",
        "content_format": "text",
        "content_hash": "hash-1",
        "author": "A. Author",
        "canonical_url": "https://a.example/1",
        "change_percentage": 12.5,
        "diff_summary": "+2 -1",
        "change_type": "content",
        "alert_matches": [7],
    }
    with db.transaction() as conn:
        persist_subscription_item(conn, source_id, item, run_id=42, now="2026-07-25T00:00:00Z")

    row = db.conn.execute(
        "SELECT content, content_kind, content_format, status, run_id, alert_matches, "
        "canonical_url, change_percentage, diff_summary, change_type "
        "FROM subscription_items WHERE url = ?",
        ("https://a.example/1",),
    ).fetchone()

    assert row[0] == "retrieval quality rubric"
    assert row[1] == "article"
    assert row[2] == "text"
    assert row[3] == "new"
    assert row[4] == 42
    assert row[5] == "[7]"
    assert row[6] == "https://a.example/1"
    assert row[7] == 12.5
    assert row[8] == "+2 -1"
    assert row[9] == "content"


def test_upsert_preserves_reviewed_status(db, source_id):
    item = {"url": "https://a.example/1", "title": "T", "content_hash": "h", "content": "body"}
    with db.transaction() as conn:
        persist_subscription_item(conn, source_id, item, run_id=1, now="2026-07-25T00:00:00Z")
        conn.execute("UPDATE subscription_items SET status = 'reviewed' WHERE url = ?",
                     ("https://a.example/1",))
        persist_subscription_item(conn, source_id, item, run_id=2, now="2026-07-25T01:00:00Z")

    row = db.conn.execute(
        "SELECT status, run_id FROM subscription_items WHERE url = ?", ("https://a.example/1",)
    ).fetchone()
    assert row[0] == "reviewed"
    assert row[1] == 2


def test_rejects_invalid_kind_format_pairing(db, source_id):
    item = {
        "url": "https://a.example/1",
        "title": "T",
        "content_hash": "h",
        "content_kind": "change",
        "content_format": "markdown",
    }
    with pytest.raises(ValueError, match="content_kind"):
        with db.transaction() as conn:
            persist_subscription_item(conn, source_id, item, run_id=1, now="2026-07-25T00:00:00Z")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions/test_item_persist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Subscriptions.item_persist'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Subscriptions/item_persist.py`:

```python
"""Single persistence path for scraped subscription items.

Replaces two divergent INSERT statements that wrote disjoint column sets:
``Subscriptions_DB`` wrote the change/dedup fields but dropped run linkage and
status, while ``LocalWatchlistsService`` did the reverse. Neither wrote body
text. Every caller now routes through :func:`persist_subscription_item`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


_VALID_PAIRINGS = {
    ("article", "text"),
    ("article", "markdown"),
    ("change", "diff"),
}


def _json_or_none(value: Any) -> str | None:
    return json.dumps(value) if value is not None else None


def _validate_content_pairing(kind: Any, fmt: Any) -> None:
    """Reject impossible kind/format combinations at the persist boundary."""
    if kind is None and fmt is None:
        return
    if (kind, fmt) not in _VALID_PAIRINGS:
        raise ValueError(
            f"invalid content_kind/content_format pairing: {kind!r}/{fmt!r}. "
            f"Valid pairings: {sorted(_VALID_PAIRINGS)}"
        )


def persist_subscription_item(
    conn: Any,
    subscription_id: int,
    item: Mapping[str, Any],
    run_id: int | None,
    now: str,
) -> None:
    """Insert or update one item, writing the full column set.

    Existing ``reviewed`` and ``ignored`` statuses are preserved across
    re-fetches; anything else resets to ``new``.

    Args:
        conn: An open connection inside a transaction.
        subscription_id: Owning source id.
        item: Normalized item mapping.
        run_id: Run that produced this item, if any.
        now: ISO-8601 timestamp for created_at/updated_at.

    Raises:
        ValueError: If content_kind and content_format are an invalid pairing.
    """
    content_kind = item.get("content_kind")
    content_format = item.get("content_format")
    _validate_content_pairing(content_kind, content_format)

    conn.execute(
        """
        INSERT INTO subscription_items (
            subscription_id, url, title, content, content_kind, content_format,
            content_hash, published_date, author, categories, enclosures,
            extracted_data, status, run_id, alert_matches, canonical_url,
            previous_hash, change_percentage, diff_summary, change_type,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subscription_id, url, content_hash) DO UPDATE SET
            title = excluded.title,
            content = excluded.content,
            content_kind = excluded.content_kind,
            content_format = excluded.content_format,
            published_date = excluded.published_date,
            author = excluded.author,
            categories = excluded.categories,
            enclosures = excluded.enclosures,
            extracted_data = excluded.extracted_data,
            run_id = excluded.run_id,
            alert_matches = excluded.alert_matches,
            canonical_url = excluded.canonical_url,
            previous_hash = excluded.previous_hash,
            change_percentage = excluded.change_percentage,
            diff_summary = excluded.diff_summary,
            change_type = excluded.change_type,
            status = CASE
                WHEN subscription_items.status IN ('reviewed', 'ignored')
                THEN subscription_items.status
                ELSE 'new'
            END,
            updated_at = excluded.updated_at
        """,
        (
            subscription_id,
            item.get("url"),
            item.get("title"),
            item.get("content"),
            content_kind,
            content_format,
            item.get("content_hash"),
            item.get("published_date"),
            item.get("author"),
            _json_or_none(item.get("categories")),
            _json_or_none(item.get("enclosures")),
            _json_or_none(item.get("extracted_data")),
            "new",
            run_id,
            _json_or_none(item.get("alert_matches")),
            item.get("canonical_url"),
            item.get("previous_hash"),
            item.get("change_percentage"),
            item.get("diff_summary"),
            item.get("change_type"),
            now,
            now,
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions/test_item_persist.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Route LocalWatchlistsService through it**

In `tldw_chatbook/Subscriptions/local_watchlists_service.py`, add the import near the other local imports:

```python
from .item_persist import persist_subscription_item
```

Then replace the whole `cursor.execute("""INSERT INTO subscription_items …""", (...))` call spanning `:1301-1340` with:

```python
                persist_subscription_item(
                    conn,
                    source_id,
                    {
                        **item,
                        "alert_matches": alert_matches,
                    },
                    run_id=run_id,
                    now=now,
                )
```

The enclosing method is `_upsert_subscription_items(db, source_id, run_id, items)`, whose body is
`with db.transaction() as conn:` → `cursor = conn.cursor()` → `for item in items:`. So `conn` is in
scope. `persist_subscription_item` takes the **connection**, not the cursor — pass `conn`.

That INSERT is the last statement in the file, so several locals become dead once it is gone.
Delete all of these:

- `title = item.get("title")`
- `published_date = item.get("published_date")`
- `author = item.get("author")`
- `categories = item.get("categories")`
- `enclosures = item.get("enclosures")`
- `extracted_data = item.get("extracted_data")`
- `cursor = conn.cursor()` — nothing else in the method uses it

Keep `url`, `content_hash` (both feed the `if not url or not content_hash: continue` guard) and
`alert_matches` (passed through in the mapping above).

- [ ] **Step 6: Verify no regressions in the watchlists suites**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions -v && pytest Tests/Watchlists -v`
Expected: PASS. If a test asserts on dropped columns, that test was encoding the bug — update it to assert the full set.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Subscriptions/item_persist.py tldw_chatbook/Subscriptions/local_watchlists_service.py Tests/Subscriptions/test_item_persist.py
git commit -m "fix(watchlists): unify item persistence into one full-column path"
```

---

### Task 5: WatchlistBundleService

**Files:**
- Create: `tldw_chatbook/Subscriptions/watchlist_bundle_service.py`
- Test: `Tests/Subscriptions/test_watchlist_bundle_service.py`

**Interfaces:**
- Consumes: Task 1's `watchlists` / `watchlist_sources` tables.
- Produces: `WatchlistBundleService(db)` with `create(name, description=None, tags=None) -> dict`, `rename(watchlist_id, name) -> dict`, `delete(watchlist_id) -> None`, `list_watchlists() -> list[dict]`, `add_source(watchlist_id, subscription_id) -> None`, `remove_source(watchlist_id, subscription_id) -> None`, `list_sources(watchlist_id) -> list[int]`. Tasks 6 and 7 and all of Phase C consume these.

- [ ] **Step 1: Write the failing test**

Create `Tests/Subscriptions/test_watchlist_bundle_service.py`:

```python
import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


@pytest.fixture
def service(db):
    return WatchlistBundleService(db)


def test_create_and_list(service):
    created = service.create("Morning AI Brief", tags=["ai", "daily"])
    assert created["name"] == "Morning AI Brief"
    assert created["tags"] == ["ai", "daily"]

    listed = service.list_watchlists()
    assert [row["name"] for row in listed] == ["Morning AI Brief"]


def test_name_collision_is_case_insensitive_and_suffixes(service):
    service.create("Unsorted")
    second = service.create("unsorted")
    third = service.create("UNSORTED")
    assert second["name"] == "unsorted (2)"
    assert third["name"] == "UNSORTED (3)"


def test_rename_also_avoids_collision(service):
    service.create("Security")
    other = service.create("Papers")
    renamed = service.rename(other["id"], "security")
    assert renamed["name"] == "security (2)"


def test_membership_add_remove_and_idempotent_add(service, db):
    watchlist = service.create("Morning")
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")

    service.add_source(watchlist["id"], source_id)
    service.add_source(watchlist["id"], source_id)  # idempotent
    assert service.list_sources(watchlist["id"]) == [source_id]

    service.remove_source(watchlist["id"], source_id)
    assert service.list_sources(watchlist["id"]) == []


def test_source_can_belong_to_multiple_watchlists(service, db):
    first = service.create("Morning")
    second = service.create("Security")
    source_id = db.add_subscription(name="HN", type="rss", source="https://b.example/f")

    service.add_source(first["id"], source_id)
    service.add_source(second["id"], source_id)

    assert service.list_sources(first["id"]) == [source_id]
    assert service.list_sources(second["id"]) == [source_id]


def test_delete_removes_membership_but_not_sources(service, db):
    watchlist = service.create("Morning")
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    service.add_source(watchlist["id"], source_id)

    service.delete(watchlist["id"])

    assert service.list_watchlists() == []
    assert db.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions/test_watchlist_bundle_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Subscriptions.watchlist_bundle_service'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Subscriptions/watchlist_bundle_service.py`:

```python
"""Watchlist bundle CRUD and source membership.

A watchlist is a named bundle of sources — the unit of organization and
checking, and (in a later slice) of briefing generation. Membership is
many-to-many: a source may belong to any number of watchlists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger

from ..DB.Subscriptions_DB import SubscriptionsDB


logger = logger.bind(module="WatchlistBundleService")


class WatchlistBundleService:
    """Local watchlist bundles backed by ``SubscriptionsDB``."""

    def __init__(self, db: SubscriptionsDB) -> None:
        self._db = db

    # --- Helpers ---

    @staticmethod
    def _split_tags(raw: Any) -> list[str]:
        """Parse the comma-joined tags column used by subscriptions.tags."""
        if not raw:
            return []
        return [part.strip() for part in str(raw).split(",") if part.strip()]

    @staticmethod
    def _join_tags(tags: Sequence[str] | None) -> str | None:
        if not tags:
            return None
        cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
        return ",".join(cleaned) if cleaned else None

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "tags": WatchlistBundleService._split_tags(row[3]),
            "is_active": bool(row[4]),
            "sort_order": row[5],
        }

    def _unique_name(self, conn: Any, name: str, exclude_id: int | None = None) -> str:
        """Return ``name``, suffixed if it collides case-insensitively.

        Uniqueness lives here rather than in a SQL UNIQUE constraint because a
        constraint would raise mid-migration on case-variant folder values or
        OPML re-imports.
        """
        base = name.strip()
        params: list[Any] = []
        query = "SELECT LOWER(name) FROM watchlists"
        if exclude_id is not None:
            query += " WHERE id != ?"
            params.append(exclude_id)
        taken = {row[0] for row in conn.execute(query, params)}

        if base.lower() not in taken:
            return base
        suffix = 2
        while f"{base.lower()} ({suffix})" in taken:
            suffix += 1
        return f"{base} ({suffix})"

    def _get(self, conn: Any, watchlist_id: int) -> dict[str, Any]:
        row = conn.execute(
            "SELECT id, name, description, tags, is_active, sort_order "
            "FROM watchlists WHERE id = ?",
            (watchlist_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no watchlist with id {watchlist_id}")
        return self._row_to_dict(row)

    # --- CRUD ---

    def create(
        self,
        name: str,
        description: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Create a watchlist, auto-suffixing the name on collision."""
        with self._db.transaction() as conn:
            resolved = self._unique_name(conn, name)
            cursor = conn.execute(
                "INSERT INTO watchlists (name, description, tags) VALUES (?, ?, ?)",
                (resolved, description, self._join_tags(tags)),
            )
            return self._get(conn, cursor.lastrowid)

    def rename(self, watchlist_id: int, name: str) -> dict[str, Any]:
        """Rename a watchlist, auto-suffixing on collision with another row."""
        with self._db.transaction() as conn:
            resolved = self._unique_name(conn, name, exclude_id=watchlist_id)
            conn.execute(
                "UPDATE watchlists SET name = ? WHERE id = ?", (resolved, watchlist_id)
            )
            return self._get(conn, watchlist_id)

    def delete(self, watchlist_id: int) -> None:
        """Delete a watchlist. Membership cascades; sources are untouched."""
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))

    def list_watchlists(self) -> list[dict[str, Any]]:
        """All watchlists in display order."""
        rows = self._db.conn.execute(
            "SELECT id, name, description, tags, is_active, sort_order "
            "FROM watchlists ORDER BY sort_order, LOWER(name)"
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # --- Membership ---

    def add_source(self, watchlist_id: int, subscription_id: int) -> None:
        """Add a source to a watchlist. Idempotent."""
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist_sources (watchlist_id, subscription_id) "
                "VALUES (?, ?)",
                (watchlist_id, subscription_id),
            )

    def remove_source(self, watchlist_id: int, subscription_id: int) -> None:
        """Remove a source from a watchlist. The source itself survives."""
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM watchlist_sources "
                "WHERE watchlist_id = ? AND subscription_id = ?",
                (watchlist_id, subscription_id),
            )

    def list_sources(self, watchlist_id: int) -> list[int]:
        """Subscription ids belonging to a watchlist."""
        rows = self._db.conn.execute(
            "SELECT subscription_id FROM watchlist_sources "
            "WHERE watchlist_id = ? ORDER BY added_at, subscription_id",
            (watchlist_id,),
        ).fetchall()
        return [row[0] for row in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions/test_watchlist_bundle_service.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/watchlist_bundle_service.py Tests/Subscriptions/test_watchlist_bundle_service.py
git commit -m "feat(watchlists): add WatchlistBundleService with membership"
```

---

### Task 6: Folder migration

**Files:**
- Modify: `tldw_chatbook/Subscriptions/watchlist_bundle_service.py`
- Test: `Tests/Subscriptions/test_watchlist_bundle_service.py`

**Interfaces:**
- Consumes: Task 5's `create`/`add_source`; Task 1's `watchlist_migration_state`.
- Produces: `WatchlistBundleService.migrate_folders() -> bool` — `True` if it ran, `False` if already applied.

**Expectation-setting:** this migration will do almost nothing for real users. `subscriptions.folder` is never written by any live path — `_subscription_config_fields` (`local_watchlists_service.py:567`) allowlists ten fields and `folder` is not among them, and nothing in `Subscriptions/` calls `add_subscription(folder=…)`. It is retained for hand-seeded databases. First-run assumes an empty watchlist set; the permanent "All sources" and "Unassigned" tree roots in Phase C are what make sources reachable.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Subscriptions/test_watchlist_bundle_service.py`:

```python
MIGRATION_KEY = "folders_to_watchlists"


def test_migrate_folders_groups_by_folder(service, db):
    first = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f",
                                folder="Research")
    second = db.add_subscription(name="HN", type="rss", source="https://b.example/f",
                                 folder="Research")
    third = db.add_subscription(name="Krebs", type="rss", source="https://c.example/f",
                                folder="Security")

    assert service.migrate_folders() is True

    names = {row["name"] for row in service.list_watchlists()}
    assert names == {"Research", "Security"}

    by_name = {row["name"]: row["id"] for row in service.list_watchlists()}
    assert sorted(service.list_sources(by_name["Research"])) == sorted([first, second])
    assert service.list_sources(by_name["Security"]) == [third]


def test_migrate_folders_puts_folderless_sources_in_unsorted(service, db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")

    service.migrate_folders()

    by_name = {row["name"]: row["id"] for row in service.list_watchlists()}
    assert "Unsorted" in by_name
    assert service.list_sources(by_name["Unsorted"]) == [source_id]


def test_migrate_folders_runs_once(service, db):
    db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f", folder="Research")

    assert service.migrate_folders() is True
    assert service.migrate_folders() is False
    assert len(service.list_watchlists()) == 1

    marker = db.conn.execute(
        "SELECT COUNT(*) FROM watchlist_migration_state WHERE key = ?", (MIGRATION_KEY,)
    ).fetchone()[0]
    assert marker == 1


def test_migrate_folders_is_noop_with_no_sources(service):
    assert service.migrate_folders() is True
    assert service.list_watchlists() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions/test_watchlist_bundle_service.py -k migrate -v`
Expected: FAIL — `AttributeError: 'WatchlistBundleService' object has no attribute 'migrate_folders'`.

- [ ] **Step 3: Write the implementation**

Add to `WatchlistBundleService`, after `list_sources`:

```python
    MIGRATION_KEY = "folders_to_watchlists"

    def migrate_folders(self) -> bool:
        """Turn distinct ``subscriptions.folder`` values into watchlists.

        Sources with no folder join a single ``Unsorted`` watchlist. The
        ``folder`` column is left in place and untouched, so this is reversible.

        In practice this migrates almost nothing: no live code path writes
        ``folder``. It exists for hand-seeded databases.

        Returns:
            ``True`` if the migration ran, ``False`` if it had already been
            applied.
        """
        with self._db.transaction() as conn:
            already = conn.execute(
                "SELECT 1 FROM watchlist_migration_state WHERE key = ?",
                (self.MIGRATION_KEY,),
            ).fetchone()
            if already:
                return False

            rows = conn.execute(
                "SELECT id, folder FROM subscriptions ORDER BY id"
            ).fetchall()

            buckets: dict[str, list[int]] = {}
            for subscription_id, folder in rows:
                label = (folder or "").strip() or "Unsorted"
                buckets.setdefault(label, []).append(subscription_id)

            for label, source_ids in buckets.items():
                resolved = self._unique_name(conn, label)
                cursor = conn.execute(
                    "INSERT INTO watchlists (name) VALUES (?)", (resolved,)
                )
                watchlist_id = cursor.lastrowid
                conn.executemany(
                    "INSERT OR IGNORE INTO watchlist_sources "
                    "(watchlist_id, subscription_id) VALUES (?, ?)",
                    [(watchlist_id, source_id) for source_id in source_ids],
                )

            conn.execute(
                "INSERT INTO watchlist_migration_state (key) VALUES (?)",
                (self.MIGRATION_KEY,),
            )
            logger.info(
                "Migrated {} folder group(s) into watchlists.", len(buckets)
            )
            return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions/test_watchlist_bundle_service.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/watchlist_bundle_service.py Tests/Subscriptions/test_watchlist_bundle_service.py
git commit -m "feat(watchlists): migrate subscription folders into watchlists"
```

---

### Task 7: Single-query tree counts

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` — new `get_watchlist_item_counts` method
- Test: `Tests/DB/test_subscriptions_db_watchlists.py`

**Interfaces:**
- Consumes: Tasks 1 and 5.
- Produces: `SubscriptionsDB.get_watchlist_item_counts() -> dict[int, dict[str, int]]`, keyed by watchlist id plus two sentinels: `-1` = Unassigned, `-2` = All sources. Each value is `{"total": int, "unread": int}`. Phase C's tree consumes this.

**Why this matters:** counting per tree node would issue one query per watchlist on every refresh. No per-subscription count helper exists today — `Subscriptions_DB` has only `get_new_items` and a single `COUNT(*)` in the whole file. This repo has form here: the performance audit found the Console's 0.2s tick running SQLite on the event loop. A `UNION ALL` keeps all three bucket shapes in one statement.

- [ ] **Step 1: Write the failing test**

Append to `Tests/DB/test_subscriptions_db_watchlists.py`:

```python
class _CountingConnection:
    """Wraps a connection to count execute() calls."""

    def __init__(self, inner):
        self._inner = inner
        self.execute_count = 0

    def execute(self, *args, **kwargs):
        self.execute_count += 1
        return self._inner.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_counts_bucket_by_watchlist_unassigned_and_all(db):
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

    service = WatchlistBundleService(db)
    morning = service.create("Morning")
    security = service.create("Security")

    shared = db.add_subscription(name="HN", type="rss", source="https://b.example/f")
    lonely = db.add_subscription(name="Orphan", type="rss", source="https://c.example/f")

    service.add_source(morning["id"], shared)
    service.add_source(security["id"], shared)

    _insert_item(db, shared, "https://b.example/1", "Shared unread", "body")
    _insert_item(db, lonely, "https://c.example/1", "Orphan unread", "body")
    with db.transaction() as conn:
        conn.execute("UPDATE subscription_items SET status = 'reviewed' WHERE url = ?",
                     ("https://c.example/1",))

    counts = db.get_watchlist_item_counts()

    assert counts[morning["id"]] == {"total": 1, "unread": 1}
    assert counts[security["id"]] == {"total": 1, "unread": 1}
    assert counts[-1] == {"total": 1, "unread": 0}   # Unassigned
    assert counts[-2] == {"total": 2, "unread": 1}   # All sources


def test_counts_use_a_single_query_regardless_of_watchlist_count(db, monkeypatch):
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

    service = WatchlistBundleService(db)
    for index in range(12):
        service.create(f"List {index}")

    counting = _CountingConnection(db.conn)
    monkeypatch.setattr(type(db), "conn", property(lambda self: counting))

    db.get_watchlist_item_counts()

    assert counting.execute_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db_watchlists.py -k counts -v`
Expected: FAIL — `AttributeError: 'SubscriptionsDB' object has no attribute 'get_watchlist_item_counts'`.

- [ ] **Step 3: Write the implementation**

Add to `SubscriptionsDB`, after `backfill_items_fts`:

```python
    # Sentinel bucket ids for the watchlists tree roots.
    UNASSIGNED_BUCKET = -1
    ALL_SOURCES_BUCKET = -2

    def get_watchlist_item_counts(self) -> Dict[int, Dict[str, int]]:
        """Item totals and unread counts for every watchlists tree node.

        Returned in a single query so that adding watchlists never adds
        round-trips. ``SUM(CASE …)`` is used rather than ``COUNT(*) FILTER``
        to avoid depending on a newer SQLite than the bundled one.

        Returns:
            Mapping of bucket id to ``{"total": int, "unread": int}``. Bucket
            ``-1`` is Unassigned (sources in no watchlist) and ``-2`` is All
            sources. Real watchlist ids are positive.
        """
        rows = self.conn.execute(
            """
            SELECT ws.watchlist_id AS bucket,
                   COUNT(si.id) AS total,
                   SUM(CASE WHEN si.status = 'new' THEN 1 ELSE 0 END) AS unread
            FROM watchlist_sources ws
            JOIN subscription_items si ON si.subscription_id = ws.subscription_id
            GROUP BY ws.watchlist_id

            UNION ALL

            SELECT ?, COUNT(si.id),
                   SUM(CASE WHEN si.status = 'new' THEN 1 ELSE 0 END)
            FROM subscription_items si
            WHERE NOT EXISTS (
                SELECT 1 FROM watchlist_sources ws
                WHERE ws.subscription_id = si.subscription_id
            )

            UNION ALL

            SELECT ?, COUNT(si.id),
                   SUM(CASE WHEN si.status = 'new' THEN 1 ELSE 0 END)
            FROM subscription_items si
            """,
            (self.UNASSIGNED_BUCKET, self.ALL_SOURCES_BUCKET),
        ).fetchall()

        return {
            row[0]: {"total": row[1] or 0, "unread": row[2] or 0}
            for row in rows
            if row[0] is not None
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/DB/test_subscriptions_db_watchlists.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run the full affected suites**

Run: `source .venv/bin/activate && pytest Tests/DB Tests/Subscriptions Tests/Watchlists -v`
Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py Tests/DB/test_subscriptions_db_watchlists.py
git commit -m "feat(watchlists): add single-query tree item counts"
```

---

## Phase A Completion Checklist

- [ ] All seven tasks committed.
- [ ] `pytest Tests/DB Tests/Subscriptions Tests/Watchlists` passes.
- [ ] `_ensure_watchlists_schema` is still idempotent — run it twice against an existing DB and confirm no error.
- [ ] No UI file was modified. `git diff --stat origin/dev -- tldw_chatbook/UI` is empty.
- [ ] Phase B plan written against merged Phase A code.
