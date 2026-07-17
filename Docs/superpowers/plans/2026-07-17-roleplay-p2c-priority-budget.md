# Roleplay P2c — entry priority + priority-aware budget — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give world-book entries a `priority` so important lore survives token-budget pressure and injects first, and fix the `token_budget`/`scan_depth` floor.

**Architecture:** Add a `priority` column (schema v20→v21 migration), thread it through `WorldBookManager`, and make `world_info_processor` sort matched entries by `(-priority, insertion_order)` for both budget survival and injection order; surface priority in the diagnostics + the Lore editor/Try-it.

**Tech Stack:** Python 3.11+, Textual, SQLite (ChaChaNotes DB, `db_schema_version` table), `WorldBookManager` + `WorldInfoProcessor`, pytest.

**Spec:** `Docs/superpowers/specs/2026-07-17-roleplay-p2c-priority-budget-design.md` (`87300806`). Branch `claude/roleplay-p2c-priority-budget` off dev `5c0de75a`.

## Global Constraints

- **Migration is idempotent.** SQLite has no `ADD COLUMN IF NOT EXISTS`, so guard the ALTER with `PRAGMA table_info`; recreate triggers with `DROP TRIGGER IF EXISTS` + `CREATE` (the P1e lesson). A replayed/downgrade migration must not error.
- **Base DDL == migrated DB.** Update both the `world_book_entries` base DDL and the migration so a fresh DB and a migrated DB end identical (column + both sync triggers carrying `priority`).
- **Priority is `(-priority, insertion_order)`.** The sort key for both budget survival and injection order. Priority is int-coerced, default `0`, clamped `0–100` at the editor.
- **Nearly inert for existing data.** Every existing entry defaults to `priority = 0`, and Python's sort is stable, so at equal priorities + default `insertion_order` the sort is a no-op — the P2a budget/recursion pins hold **unchanged** (`test_diagnostics_classifies_disabled_secondary_and_budget`, `test_diagnostics_reports_recursively_fired_entries`). The one behavior change at default priority is the recursive-scan reorder (a recursively-matched entry re-orders by its own `insertion_order` instead of always trailing) — pinned by a new test.
- **`token_budget = 0` means unlimited** (pre-existing: `_apply_token_budget` early-returns when `token_budget <= 0`). The floor fix seeds `token_budget`/`scan_depth` to `0` so lower book values win `max(…)`; a book that *omits* the field still defaults to 500/3.
- **⚠️ Schema-version collision:** claim v21. Re-verify at merge time that no parallel branch also minted v21 (the v19-collision lesson).
- **Test env** (prefix every pytest run): `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`
- **Staging:** stage only each task's own files. Never `git add -A`; never stage `.superpowers/`.

## File structure

- `tldw_chatbook/DB/ChaChaNotes_DB.py` — base DDL + `_MIGRATE_V20_TO_V21_SQL` + `_migrate_from_v20_to_v21` + version bump + `migration_steps` (Task 1)
- `tldw_chatbook/DB/migrations/chachanotes_v20_to_v21_world_book_entry_priority.sql` — doc-mirror (Task 1)
- `tldw_chatbook/Character_Chat/world_book_manager.py` — priority in create/update/get/export/import (Task 2)
- `tldw_chatbook/Character_Chat/world_info_processor.py` — priority carry + `(-priority, insertion_order)` sort + floor fix (Task 3)
- `tldw_chatbook/Character_Chat/world_info_diagnostics.py` + `Widgets/Persona_Widgets/personas_lore_detail.py` + `personas_lore_tryit.py` — priority surfacing (Task 4)
- Tests: `Tests/DB/test_chachanotes_world_book_priority_migration.py` (Task 1), `Tests/Character_Chat/test_world_book_manager.py` (Task 2), `Tests/Character_Chat/test_world_info_diagnostics.py` (Task 3), `Tests/UI/test_personas_lore.py` (Task 4)

---

## Task 1: Schema migration v20→v21 (`priority` column + sync triggers)

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `tldw_chatbook/DB/migrations/chachanotes_v20_to_v21_world_book_entry_priority.sql`
- Test: `Tests/DB/test_chachanotes_world_book_priority_migration.py` (create)

**Interfaces:**
- Produces: `world_book_entries.priority INTEGER DEFAULT 0` on fresh + migrated DBs; `_CURRENT_SCHEMA_VERSION = 21`. Consumed by Tasks 2–4.

- [ ] **Step 1: Write the failing migration test.** Create `Tests/DB/test_chachanotes_world_book_priority_migration.py` (mirrors `Tests/Chat/test_conversation_local_marks_service.py::test_local_marks_migrate_from_v16_to_v17_with_expected_schema`):

```python
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_world_book_entries_priority_migrate_v20_to_v21(tmp_path):
    db_path = tmp_path / "chacha.sqlite"
    db = CharactersRAGDB(str(db_path), client_id="test-client")
    conn = db.get_connection()
    # Simulate a V20-shaped DB: drop the V21 additions.
    conn.execute("DROP TRIGGER IF EXISTS world_book_entries_sync_create")
    conn.execute("DROP TRIGGER IF EXISTS world_book_entries_sync_update")
    conn.execute("ALTER TABLE world_book_entries DROP COLUMN priority")
    conn.execute(
        "UPDATE db_schema_version SET version = 20 WHERE schema_name = ?",
        (db._SCHEMA_NAME,),
    )
    conn.commit()
    db.close_connection()

    # Reopen → auto-migrates V20→V21.
    migrated = CharactersRAGDB(str(db_path), client_id="test-client")
    mconn = migrated.get_connection()
    version = mconn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (migrated._SCHEMA_NAME,),
    ).fetchone()
    assert version["version"] == migrated._CURRENT_SCHEMA_VERSION == 21
    cols = {r[1] for r in mconn.execute("PRAGMA table_info(world_book_entries)").fetchall()}
    assert "priority" in cols
    create_sql = mconn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_create'"
    ).fetchone()["sql"]
    assert "priority" in create_sql
    update_sql = mconn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_update'"
    ).fetchone()["sql"]
    assert "priority" in update_sql


def test_fresh_db_has_priority_column_and_triggers(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "fresh.sqlite"), client_id="test-client")
    conn = db.get_connection()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(world_book_entries)").fetchall()}
    assert "priority" in cols
    create_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_create'"
    ).fetchone()["sql"]
    assert "priority" in create_sql
```

- [ ] **Step 2: Run — expect FAIL** (fresh DB is still v20; no `priority`).

Run: `... -m pytest Tests/DB/test_chachanotes_world_book_priority_migration.py -q`

- [ ] **Step 3: Implement.** In `ChaChaNotes_DB.py`:
  - **Base DDL** — in the `world_book_entries` `CREATE TABLE` (~`:1194`), add `priority INTEGER DEFAULT 0,` after `insertion_order INTEGER DEFAULT 0,`. In the base `world_book_entries_sync_create` trigger's `json_object(...)` add `'priority', NEW.priority,` (before `'created_at'`). In `world_book_entries_sync_update`: add `OR OLD.priority IS NOT NEW.priority` to the `WHEN` clause, and `'priority', NEW.priority,` to its `json_object(...)`.
  - **Version** — bump `_CURRENT_SCHEMA_VERSION = 21` (`:143`).
  - **Migration SQL constant** — add near the other `_MIGRATE_*` constants:

```python
    _MIGRATE_V20_TO_V21_SQL = """
DROP TRIGGER IF EXISTS world_book_entries_sync_create;
DROP TRIGGER IF EXISTS world_book_entries_sync_update;

CREATE TRIGGER world_book_entries_sync_create
AFTER INSERT ON world_book_entries BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('world_book_entries', CAST(NEW.id AS TEXT), 'create', NEW.last_modified,
         (SELECT client_id FROM world_books WHERE id = NEW.world_book_id), 1,
         json_object('id', NEW.id, 'world_book_id', NEW.world_book_id, 'keys', NEW.keys,
                     'content', NEW.content, 'enabled', NEW.enabled, 'position', NEW.position,
                     'insertion_order', NEW.insertion_order, 'priority', NEW.priority,
                     'selective', NEW.selective, 'secondary_keys', NEW.secondary_keys,
                     'case_sensitive', NEW.case_sensitive, 'extensions', NEW.extensions,
                     'created_at', NEW.created_at, 'last_modified', NEW.last_modified));
END;

CREATE TRIGGER world_book_entries_sync_update
AFTER UPDATE ON world_book_entries
WHEN OLD.keys IS NOT NEW.keys OR
     OLD.content IS NOT NEW.content OR
     OLD.enabled IS NOT NEW.enabled OR
     OLD.position IS NOT NEW.position OR
     OLD.insertion_order IS NOT NEW.insertion_order OR
     OLD.priority IS NOT NEW.priority OR
     OLD.selective IS NOT NEW.selective OR
     OLD.secondary_keys IS NOT NEW.secondary_keys OR
     OLD.case_sensitive IS NOT NEW.case_sensitive OR
     OLD.extensions IS NOT NEW.extensions
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('world_book_entries', CAST(NEW.id AS TEXT), 'update', NEW.last_modified,
         (SELECT client_id FROM world_books WHERE id = NEW.world_book_id), 1,
         json_object('id', NEW.id, 'world_book_id', NEW.world_book_id, 'keys', NEW.keys,
                     'content', NEW.content, 'enabled', NEW.enabled, 'position', NEW.position,
                     'insertion_order', NEW.insertion_order, 'priority', NEW.priority,
                     'selective', NEW.selective, 'secondary_keys', NEW.secondary_keys,
                     'case_sensitive', NEW.case_sensitive, 'extensions', NEW.extensions,
                     'created_at', NEW.created_at, 'last_modified', NEW.last_modified));
END;

UPDATE db_schema_version SET version = 21 WHERE schema_name = 'rag_char_chat_schema';
"""
```

  - **Migration method** — add (mirroring `_migrate_from_v19_to_v20` at `:3256`):

```python
    def _migrate_from_v20_to_v21(self, conn: sqlite3.Connection):
        """Migrate schema V20 → V21: add ``world_book_entries.priority`` (entry
        injection priority / budget-survival weight) and redefine the two
        ``world_book_entries_sync_*`` triggers so the new column is synced."""
        logger.info(f"Migrating schema from V20 to V21 for '{self._SCHEMA_NAME}' in DB: {self.db_path_str}...")
        try:
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(world_book_entries)").fetchall()
            }
            if "priority" not in existing_columns:
                conn.execute("ALTER TABLE world_book_entries ADD COLUMN priority INTEGER DEFAULT 0")
            conn.executescript(self._MIGRATE_V20_TO_V21_SQL)
            final_version = self._get_db_version(conn)
            if final_version != 21:
                raise SchemaError(
                    f"[{self._SCHEMA_NAME} V20→V21] Migration version check failed. Expected 21, got: {final_version}"
                )
            logger.info(f"[{self._SCHEMA_NAME} V20→V21] Migration completed for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logger.opt(exception=True).error(f"[{self._SCHEMA_NAME} V20→V21] Migration failed: {e}")
            raise SchemaError(f"Migration from V20 to V21 failed for '{self._SCHEMA_NAME}': {e}") from e
```

  - **Register** — in the `migration_steps` dict (`:3390`), add `20: self._migrate_from_v20_to_v21,` after the `19:` entry.
  - **Doc-mirror** — create `tldw_chatbook/DB/migrations/chachanotes_v20_to_v21_world_book_entry_priority.sql` containing the `ALTER TABLE world_book_entries ADD COLUMN priority INTEGER DEFAULT 0;` line + the `_MIGRATE_V20_TO_V21_SQL` body (for documentation; the code path runs the constant).

- [ ] **Step 4: Run — PASS** (2 tests). Then the world-book/DB regression:
  `... -m pytest Tests/DB/test_chachanotes_world_book_priority_migration.py Tests/Character_Chat/test_world_book_manager.py -q`

- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/DB/migrations/chachanotes_v20_to_v21_world_book_entry_priority.sql Tests/DB/test_chachanotes_world_book_priority_migration.py
git commit -m "feat(lore): schema v20->v21 — world_book_entries.priority + synced triggers"
```

---

## Task 2: `WorldBookManager` — carry `priority`

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_book_manager.py`
- Test: `Tests/Character_Chat/test_world_book_manager.py` (add priority round-trip tests)

**Interfaces:**
- Consumes: the `priority` column (Task 1).
- Produces: `create_world_book_entry(..., priority: int = 0)`, `update_world_book_entry(**kwargs)` accepting `priority`, `get_world_book_entries` rows include `priority`, `export_world_book`/`import_world_book` carry `priority`.

- [ ] **Step 1: Write the failing tests.** Add to `Tests/Character_Chat/test_world_book_manager.py`:

```python
def test_entry_priority_round_trips(wb_manager):
    book_id = wb_manager.create_world_book("B")
    eid = wb_manager.create_world_book_entry(book_id, ["Warden"], "grim jailer", priority=75)
    assert wb_manager.get_world_book_entries(book_id)[0]["priority"] == 75
    wb_manager.update_world_book_entry(eid, priority=20)
    assert wb_manager.get_world_book_entries(book_id)[0]["priority"] == 20


def test_entry_priority_default_zero(wb_manager):
    book_id = wb_manager.create_world_book("B")
    wb_manager.create_world_book_entry(book_id, ["k"], "c")
    assert wb_manager.get_world_book_entries(book_id)[0]["priority"] == 0


def test_export_import_preserves_priority(wb_manager):
    book_id = wb_manager.create_world_book("B")
    wb_manager.create_world_book_entry(book_id, ["Warden"], "grim jailer", priority=90)
    data = wb_manager.export_world_book(book_id)
    assert data["entries"][0]["priority"] == 90
    new_id = wb_manager.import_world_book(data, name_override="B copy")
    assert wb_manager.get_world_book_entries(new_id)[0]["priority"] == 90
```

(These use the module's existing `wb_manager` fixture — a `WorldBookManager` over a real `CharactersRAGDB` — at `test_world_book_manager.py:35`. Add them as module-level functions or as `TestWorldBookManager` methods; the fixture resolves either way.)

- [ ] **Step 2: Run — expect FAIL** (`TypeError`/missing `priority` key).

- [ ] **Step 3: Implement** in `world_book_manager.py`:
  - `create_world_book_entry`: add `priority: int = 0` to the signature; add `priority` to the INSERT column list and a `?`; pass `int(priority)` in the params tuple (position matching the column order).
  - `update_world_book_entry`: add `priority` to the list of allowed update fields (it's an int, not JSON — coerce `int(kwargs["priority"])`).
  - `get_world_book_entries`: add `priority` to the `SELECT` column list and `'priority': row[<index>]` to the row dict (respect the new column's position in the SELECT).
  - `export_world_book`: add `'priority': entry['priority']` to the per-entry dict.
  - `import_world_book`: pass `priority=entry.get('priority', 0)` when creating each entry.

- [ ] **Step 4: Run — PASS.** Then `... -m pytest Tests/Character_Chat/test_world_book_manager.py -q`.

- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/Character_Chat/world_book_manager.py Tests/Character_Chat/test_world_book_manager.py
git commit -m "feat(lore): WorldBookManager carries entry priority (CRUD + export/import)"
```

---

## Task 3: `world_info_processor` — priority-aware order + floor fix

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_info_processor.py`
- Test: `Tests/Character_Chat/test_world_info_diagnostics.py`

**Interfaces:**
- Consumes: entries carrying `priority` (Task 2). Produces: processed entries + candidates tagged with `priority`; matched entries ordered by `(-priority, insertion_order)` before budget + injection.

- [ ] **Step 1: Write the failing tests.** Add to `Tests/Character_Chat/test_world_info_diagnostics.py`:

```python
def test_high_priority_entry_survives_budget_over_low_priority():
    """Priority (not FIFO) decides budget survival: the high-priority entry
    fires even though a low-priority one comes first by insertion_order."""
    book = _book(1, "B", [
        _entry(1, ["low"], "AAAA " * 200, insertion_order=0, priority=0),    # ~250 tok, first by order
        _entry(2, ["high"], "BBBB " * 200, insertion_order=1, priority=90),  # ~250 tok, high priority
    ], token_budget=300)   # fits exactly one entry, not two
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("low high", [])
    fired = result["injections"]["before_char"]
    assert any("BBBB" in c for c in fired)   # high-priority survives
    assert all("AAAA" not in c for c in fired)  # low-priority dropped


def test_low_book_token_budget_is_honored():
    """A book token_budget below the old 500 default is honored (floor fix)."""
    book = _book(1, "B", [
        _entry(1, ["a"], "AAAA " * 30, insertion_order=0),   # ~30 tok
        _entry(2, ["b"], "BBBB " * 30, insertion_order=1),   # ~30 tok, over a 40-token budget
    ], token_budget=40)
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("a b", [])
    fired = result["injections"]["before_char"]
    assert any("AAAA" in c for c in fired) and all("BBBB" not in c for c in fired)


def test_injection_order_reflects_priority():
    book = _book(1, "B", [
        _entry(1, ["a"], "content-a", insertion_order=0, priority=1),
        _entry(2, ["b"], "content-b", insertion_order=1, priority=99),
    ])
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("a b", [])
    injected = result["injections"]["before_char"]
    assert injected == ["content-b", "content-a"]   # priority 99 before priority 1


def test_recursive_entry_reorders_by_insertion_order():
    """Recursive-scan normalization: an entry that fires via recursion but has
    a LOWER insertion_order than a direct match now orders ahead of it."""
    book = _book(1, "B", [
        _entry(1, ["castle"], "the castle guards a dragon", insertion_order=5),
        _entry(2, ["dragon"], "a fearsome dragon", insertion_order=0),
    ], recursive_scanning=True)
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("the castle", [])
    injected = result["injections"]["before_char"]
    # entry 2 (insertion_order 0, fired via recursion) now precedes entry 1 (order 5).
    assert injected.index("a fearsome dragon") < injected.index("the castle guards a dragon")
```

Ensure the `_entry` helper accepts `priority` (it uses `**kw` — pass `priority=…`), and `_book` accepts `token_budget`/`recursive_scanning` (it already does via `**kw`).

- [ ] **Step 2: Run — expect FAIL** (FIFO/floor behavior).

- [ ] **Step 3: Implement** in `world_info_processor.py`:
  - `__init__`: change `self.scan_depth = 3` → `self.scan_depth = 0` and `self.token_budget = 500` → `self.token_budget = 0` (so `_process_world_books`'s `max(…)` honors lower book values; `_process_character_book` assigns directly).
  - `_process_entry`: add `'priority': _coerce_int(entry.get('priority', 0))` to the returned dict (add a small local `_coerce_int(v, default=0)` helper or inline `int(entry.get('priority') or 0)` in a try/except → default 0).
  - `process_messages`: right after `matched_entries = self._find_matching_entries(scan_text)`, add:
    ```python
    matched_entries = sorted(
        matched_entries,
        key=lambda e: (-int(e.get('priority', 0) or 0), e.get('insertion_order', 0)),
    )
    ```
    (This precedes both `_apply_token_budget` and `_organize_by_position`.) Update the stale `# Sort by insertion order (already done in initialization)` comment in `_apply_token_budget`.

- [ ] **Step 4: Run — PASS** (4 new). Then confirm the P2a equal-priority pins still pass:
  `... -m pytest Tests/Character_Chat/test_world_info_diagnostics.py Tests/Character_Chat/test_world_info.py -q`
  Expected: all pass (including `test_diagnostics_classifies_disabled_secondary_and_budget`, `test_diagnostics_reports_recursively_fired_entries`, `test_diagnostics_result_equals_plain_process_messages` — the stable-no-op property).

- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/Character_Chat/world_info_processor.py Tests/Character_Chat/test_world_info_diagnostics.py
git commit -m "feat(lore): priority-aware budget survival + injection order + budget/scan floor fix"
```

---

## Task 4: Diagnostics + Lore editor/Try-it surface `priority`

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_info_diagnostics.py`, `tldw_chatbook/Character_Chat/world_info_processor.py` (populate), `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py`, `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_tryit.py`
- Test: `Tests/Character_Chat/test_world_info_diagnostics.py`, `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: `priority` on candidates (Task 3). Produces: `WorldBookEntryDiagnostic.priority`; the editor's `#personas-lore-entry-priority` field + `entry_form_payload()["priority"]`; Try-it fired rows showing priority.

- [ ] **Step 1: Write the failing tests.** Add a diagnostics test:

```python
def test_diagnostic_record_carries_priority():
    book = _book(1, "B", [_entry(1, ["Warden"], "grim jailer", priority=42)])
    proc = WorldInfoProcessor(world_books=[book])
    _result, diag = proc.process_messages_with_diagnostics("The Warden.", [])
    fired = next(e for e in diag.entries if e.status == "fired")
    assert fired.priority == 42
    assert fired.to_dict()["priority"] == 42
```

And a widget test (in `Tests/UI/test_personas_lore.py`):

```python
@pytest.mark.asyncio
async def test_entry_priority_round_trips_through_form():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "Warden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        app.query_one("#personas-lore-entry-priority", Input).value = "80"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert app.posted[-1]["priority"] == 80
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: priority` / `NoMatches`).

- [ ] **Step 3: Implement.**
  - `world_info_diagnostics.py`: add `priority: int = 0` to `WorldBookEntryDiagnostic` (after `keys`), and `"priority": self.priority,` to its `to_dict()`.
  - `world_info_processor.py` `process_messages_with_diagnostics`: in BOTH the fired-record and near-miss-record `WorldBookEntryDiagnostic(...)` constructions, add `priority=int(cand.get('priority', 0) or 0),`.
  - `personas_lore_detail.py`:
    - In the Entries form row, add `yield Input(placeholder="Priority", id="personas-lore-entry-priority", value="0")`.
    - `entry_form_payload()`: read priority — `raw_pri = self.query_one("#personas-lore-entry-priority", Input).value.strip() or "0"; try: priority = max(0, min(100, int(raw_pri))) except ValueError: priority = 0`; include `"priority": priority` in the returned dict.
    - `on_mount` columns: add `"priority"` to `table.add_columns(...)`.
    - `update_entries`: add a `Text(str(entry.get("priority") or 0), style=style)` cell (raw string in a `Text`, per the P2a no-double-escape lesson).
    - `_fill_form_from_entry`: set `#personas-lore-entry-priority` to `str(entry.get("priority") or 0)`.
  - `personas_lore_tryit.py` `_render_diagnostics`: in the fired-row string, include priority, e.g. `f"{', '.join(keys)} → {content_preview} · pri {int(e.get('priority', 0) or 0)} · {token_cost} tok"`.

- [ ] **Step 4: Run — PASS.** Then `... -m pytest Tests/UI/test_personas_lore.py Tests/Character_Chat/test_world_info_diagnostics.py -q`.

- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/Character_Chat/world_info_diagnostics.py tldw_chatbook/Character_Chat/world_info_processor.py tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py tldw_chatbook/Widgets/Persona_Widgets/personas_lore_tryit.py Tests/Character_Chat/test_world_info_diagnostics.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): surface entry priority in diagnostics + Lore editor + Try-it"
```

---

## Task 5: Full gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-17-roleplay-p2c-priority-budget-design.md` (status line)

- [ ] **Step 1: Full gate.**
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Character_Chat/ Tests/DB/test_chachanotes_world_book_priority_migration.py \
  Tests/UI/test_personas_lore.py Tests/UI/test_personas_dictionaries.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass (record counts). Then the import smoke:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('IMPORT OK')"
```

- [ ] **Step 2: Flip spec status** — `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P2c).`

- [ ] **Step 3: Commit.**
```bash
git add Docs/superpowers/specs/2026-07-17-roleplay-p2c-priority-budget-design.md
git commit -m "docs(roleplay): mark P2c priority+budget spec implemented"
```

---

## Notes for the executor

- **Load-bearing tests:** Task 1 migration (idempotent + downgrade-replay: column + triggers present, re-run no-op); Task 3 priority-survival budget (high-priority fires over a first-by-order low-priority one) + the floor fix (book budget < 500 honored) + injection-order-by-priority + the recursive-reorder pin; and the **P2a equal-priority pins still passing unchanged** (the stable-no-op property — do NOT rewrite them).
- **The change is opt-in.** At default priority (0) + default `insertion_order`, the sort is a stable no-op — existing behavior is preserved. Behavior changes only when a user sets differing priorities or a book uses an explicit low budget/scan-depth (and the recursive-reorder edge).
- **Base DDL and migration must end identical** — a fresh DB (base DDL) and a migrated V20 DB must both have the `priority` column and the two sync triggers carrying it.
- **Markup-safety** (Task 4 DataTable cell): pass the priority string raw into a `rich.text.Text` — do NOT `escape()` (the P2a lesson).
- **Scope:** priority + budget only. No regex/selective/secondary editing, no attach, no native-send (deferred to P2d–g).
