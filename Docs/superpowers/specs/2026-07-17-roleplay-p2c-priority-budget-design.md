# Roleplay P2c — entry priority + priority-aware token budget (Lore)

**Status:** Implemented (P2c).

**Program:** Roleplay (Personas) redesign — P2 (Lore mode), third sub-project. Mirrors P1c (dictionary entry priority). Follows P2a (Lore foundation + diagnostics, merged PR #673).

## Why

P2a's diagnostics exposed two budget-correctness gaps in `world_info_processor` (also flagged by Qodo on #673):

1. **FIFO hard-break** — `_apply_token_budget` walks matched entries and `break`s at the first over-budget one, dropping *every* remaining entry regardless of importance.
2. **Budget/scan-depth floor** — `WorldInfoProcessor.__init__` seeds `token_budget=500`/`scan_depth=3` and only raises them via `max(…)`, so a book configured with *lower* values is silently floored (a Settings field that doesn't take effect).

P2c gives entries a `priority` so important lore survives budget pressure and injects first, and fixes the floor so a book's configured budget/scan-depth is honoured.

## Scope

**In scope:** entry-level `priority` (schema + manager + processor + diagnostics + editor), a priority-aware budget/injection order, and the floor fix.

**Deferred (later P2 sub-projects):** regex + selective/secondary/case editing (P2d); import/export UI (P2d — though export/import *serialization* of priority is done here so Duplicate doesn't drop it); conversation attach (P2e); character attach (P2f); Console "what's in play" + native-Console send + retiring legacy world-book UIs + the conversation-only legacy-send-bug (P2g). Constant/sticky entries: never.

## Behavior-change framing (important)

Unlike P2a (which held the legacy send byte-identical), P2c *may* change the live legacy send (`chat_events.py:1000`) — but the change is **opt-in and nearly inert for existing data**:

- Every existing entry gets `priority = 0` (column default). The new sort key is `(-priority, insertion_order)`; Python's sort is **stable**, so with all-equal priorities and the entries already in `insertion_order`, the sort is a **no-op** — output is unchanged for existing data.
- **Recursive-scan normalization (the one possible order change at default priority):** today `_find_matching_entries` *appends* recursively-matched entries after the direct matches. The new global sort re-orders all matched entries by `(-priority, insertion_order)`, so a recursively-matched entry re-orders by its own `insertion_order` instead of always trailing. This only *visibly* changes output when a recursively-matched entry has a **lower `insertion_order` than a directly-matched one** (uncommon); it is accepted as more consistent (order by the entry, not by how it matched).

So P2a's existing budget/order pins hold **unchanged** — they use default `insertion_order`, where the sort is a stable no-op (verified against `test_diagnostics_classifies_disabled_secondary_and_budget` and `test_diagnostics_reports_recursively_fired_entries`). P2c is almost purely **additive** tests; it adds one test pinning the recursive-normalization case explicitly.

## Ground truths (verified at dev `5c0de75a`)

- **Schema:** `ChaChaNotes_DB._CURRENT_SCHEMA_VERSION = 20`. `world_book_entries` base DDL (`:1194`) has `insertion_order INTEGER DEFAULT 0`, **no `priority`**. The migration system: `migration_steps` dict + per-step methods like `_migrate_from_v19_to_v20` (PRAGMA-`table_info` guard + `executescript`).
- **Sync triggers:** `world_book_entries_sync_create` and `world_book_entries_sync_update` build a `json_object(...)` payload enumerating the entry's columns explicitly, and `sync_update`'s `WHEN OLD.x IS NOT NEW.x` clause enumerates them too. The FTS triggers (`world_book_entries_ai/au/ad`) index only `keys`/`content`. So a new column needs the two **sync** triggers recreated (payload + WHEN), not the FTS ones.
- **Budget:** `_apply_token_budget` (`world_info_processor.py`) is a walk-and-stop with a hard `break`. `_process_world_books` does `self.token_budget = max(self.token_budget, book.get('token_budget', 500))` (init 500 → floor); same for `scan_depth` (init 3). `_apply_token_budget` early-returns when `token_budget <= 0` (so `0` = unlimited — pre-existing).
- **P1c mirror:** `ChatDictionary.priority` (int, `_coerce_int` default 0); `enforce_token_budget` (`Chat_Dictionary_Lib.py:504`) is *also* a walk-and-stop hard `break` — P1c's fix was purely the ordering *before* the walk. `WorldBookManager` (`create/update/get_world_book_entry`, `export/import_world_book`) has no `priority` today.

## Architecture

### 1. Schema migration v20→v21 (`ChaChaNotes_DB.py`)

- Add `_MIGRATE_V20_TO_V21_SQL` and `_migrate_from_v20_to_v21(conn)`:
  - `ALTER TABLE world_book_entries ADD COLUMN priority INTEGER DEFAULT 0`, guarded by a `PRAGMA table_info(world_book_entries)` check (SQLite has no `ADD COLUMN IF NOT EXISTS`; idempotent so a re-run/downgrade-replay is safe — the P1e lesson).
  - `DROP TRIGGER IF EXISTS world_book_entries_sync_create; CREATE TRIGGER …` and same for `world_book_entries_sync_update`, adding `'priority', NEW.priority` to both payloads and `OR OLD.priority IS NOT NEW.priority` to the update `WHEN` clause.
- Bump `_CURRENT_SCHEMA_VERSION = 21`; register `20: self._migrate_from_v20_to_v21` in `migration_steps`.
- Update the **base DDL** (`world_book_entries` table + the two sync triggers) so a fresh DB matches a migrated one exactly.
- Mirror to `tldw_chatbook/DB/migrations/chachanotes_v20_to_v21_world_book_entry_priority.sql` (doc-mirror, matching the `chachanotes_v19_to_v20_conversation_metadata.sql` convention). **Next ChaChaNotes migration = v21→v22.** Test by mirroring the existing migration-test precedent (`Tests/DB/…::test_existing_database_migration` / the P1e v19→v20 downgrade-replay test): assert the column + recreated triggers exist post-migration and that re-running the step is a no-op.

### 2. `WorldBookManager` (`world_book_manager.py`)

- `create_world_book_entry(..., priority: int = 0)` and `update_world_book_entry(**kwargs)` accept `priority` (int-coerced, default 0); the INSERT/UPDATE write it.
- `get_world_book_entries` returns `priority` in each row dict.
- `export_world_book` serializes `priority` per entry; `import_world_book` reads it (default 0 when absent) — so P2a's Duplicate (export+import) preserves priorities.

### 3. `world_info_processor.py`

- `_process_entry` carries `priority` (from the entry, `int`-coerced, default 0) in the processed dict; `_make_candidate` therefore tags candidates with it too.
- **Priority-aware order (survival + injection):** in `process_messages`, after `_find_matching_entries`, sort the matched list by `(-int(priority or 0), insertion_order)` before `_apply_token_budget` *and* `_organize_by_position`. The budget walk-and-stop then keeps the highest-priority entries, and `_organize_by_position` injects survivors in that same order.
- **Floor fix:** `__init__` seeds `token_budget = 0` and `scan_depth = 0` (not 500/3), so `_process_world_books`'s `max(…)` honours a book's lower values. (`_process_character_book` already assigns directly; a book that *omits* the field still defaults to 500/3, and `token_budget = 0` still means "no cap".)

### 4. Diagnostics (`world_info_diagnostics.py` + processor)

- `WorldBookEntryDiagnostic` gains a `priority: int = 0` field (in `to_dict()`); `process_messages_with_diagnostics` populates it from the candidate. The fired list's `injection_order` already reflects the new priority order (fired is derived from the priority-sorted `result["matched_entries"]`).
- The `result == plain` pin still holds by construction (diagnostics uses the plain path).

### 5. UI

- **`PersonasLoreDetailWidget`** — an entry priority field `#personas-lore-entry-priority` (`Input`, default "0", int-coerced 0–100 clamp); `entry_form_payload()` includes `priority`; a `priority` column in the entries `DataTable`; `_fill_form_from_entry` populates it.
- **`PersonasLoreTryItWidget`** — the fired-list row shows priority, e.g. `"{keys} → {preview} · pri {priority} · {tok} tok"`.

## Data flow (send / Try-it)

matched entries → **sort by (−priority, insertion_order)** → `_apply_token_budget` (walk-and-stop keeps highest-priority) → `_organize_by_position` (inject survivors in that order). The diagnostics path runs the same `process_messages` and additionally tags each candidate with its priority.

## Error handling

- Priority coercion never raises: non-int / null → 0 (mirrors `_coerce_int`).
- The migration is idempotent (PRAGMA guard + `DROP … IF EXISTS`); a partial/re-run leaves a consistent schema.
- The processor changes preserve the never-raise contract (a bad entry is skipped, never crashes a send or Try-it).

## Testing

- **Migration:** v20→v21 adds the `priority` column (default 0 on existing rows) and recreates the two sync triggers with `priority` in the payload; idempotent (re-running the step is a no-op); a downgrade-replay test (column already present) doesn't error.
- **Manager:** create/get/update round-trip `priority`; export includes it; import restores it (and defaults to 0 when absent).
- **Processor (load-bearing):** a high-priority entry survives budget pressure over a low-priority one that a FIFO order would have kept (the core fix); a book with `token_budget < 500` is honoured (floor fix); a book with `scan_depth < 3` is honoured; injection order reflects priority; **equal-priority behavior matches today** (the P2a pins) except the **recursive-scan normalization**, which gets its own updated expectation.
- **Diagnostics:** `priority` surfaced in the record; fired order reflects priority.
- **UI:** the priority field round-trips through `entry_form_payload`/`_fill_form_from_entry`; the Try-it fired row shows priority.
- Full gate (Character_Chat + world-info + ChaChaNotes DB migration tests + personas UI) + `import tldw_chatbook.app`.

## Acceptance criteria

- [ ] `world_book_entries` has a `priority INTEGER DEFAULT 0` column on both fresh (base DDL) and migrated (v20→v21) DBs, with the two sync triggers carrying `priority`; the migration is idempotent.
- [ ] `WorldBookManager` create/update/get/export/import carry `priority`.
- [ ] A higher-priority entry survives token-budget pressure over a lower-priority one; a book's `token_budget`/`scan_depth` below the old defaults is honoured; survivors inject in `(-priority, insertion_order)` order.
- [ ] Equal-priority (default) data reproduces today's send output, except recursively-matched entries now order by `(-priority, insertion_order)` (pinned).
- [ ] `WorldBookEntryDiagnostic` exposes `priority`; the Detail editor edits it; the Try-it fired list shows it.
- [ ] No regex/selective editing, no attach, no native-send (deferred).
- [ ] Full gate green; `import tldw_chatbook.app` OK.
