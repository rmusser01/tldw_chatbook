# Console Conversation Branching — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the native Console persist regenerated assistant replies as real message-tree
siblings, let `<` / `>` swipe navigate those persisted siblings, and reconstruct the branch you
left on resume — driven by a single per-conversation active-leaf pointer.

**Architecture:** The in-memory `ConsoleChatStore` becomes the source of truth for a conversation
*tree* (nodes linked by `parent_message_id`), with the existing flat `_messages_by_session` list
demoted to a *materialized view of the active path* (the ancestry of one active-leaf node). Normal
sends append a child of the active leaf and advance the pointer — byte-identical behavior for
linear chats. Regenerate creates a **new sibling node** under the anchor's parent and moves the
pointer onto it (mid-conversation regenerate therefore forks and truncates the visible tail, which
is preserved off-path). Writes go through to ChaChaNotes with a real `parent_message_id` (today it
is always `NULL`); the active-leaf pointer is a new **local-only** `conversations` column that never
syncs. Resume walks the active-leaf's ancestry instead of blindly following `children[-1]`.

**Tech Stack:** Python ≥3.11, Textual, SQLite (ChaChaNotes schema, FTS5 + sync triggers), pytest.

## Global Constraints

- **Spec:** `Docs/superpowers/specs/2026-07-22-console-conversation-branching-foundation-design.md`. This plan is **Phase A** of that spec's three phases; do not implement Phase B (user-message "Edit & resend" branching) or Phase C (agent-marker anchoring) here.
- **Schema:** `_CURRENT_SCHEMA_VERSION` is currently **22**; this plan takes it to **23**. The canonical `conversations` CREATE TABLE is frozen at v4 — **never edit it**; add columns only via a migration step.
- **The active-leaf pointer is local-only and MUST NOT sync.** Write it with a bare `UPDATE` that does **not** bump `version`/`last_modified` and does **not** touch any column in the `conversations_sync_update` trigger's `WHEN` clause, so no `sync_log` row is produced. Do **not** redefine the `conversations_sync_*` triggers. Do **not** route it through `update_conversation`.
- **Tests:** real in-memory SQLite (`CharactersRAGDB(":memory:", client_id=...)`), not mocks, for any DB-touching test. Run via the worktree venv: `source .venv/bin/activate && python -m pytest <path> -v`.
- **Persistence is optional.** Every store change must keep working when `self.persistence is None` (pure in-memory mode) — branch state lives in memory; only the write-through steps are skipped.
- **Style:** type hints on public APIs; Google-style docstrings; early returns; parameterized SQL only; follow the surrounding code's idioms.
- **Commit** after each task's tests pass. End commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- **Modify** `tldw_chatbook/DB/ChaChaNotes_DB.py` — v22→v23 migration (guarded ALTER + version bump only) and two accessors `set_conversation_active_leaf` / `get_conversation_active_leaf`.
- **Create** `tldw_chatbook/DB/migrations/chachanotes_v22_to_v23_conversation_active_leaf.sql` — reference mirror of the runner SQL.
- **Modify** `tldw_chatbook/Chat/console_chat_models.py` — add `parent_message_id: str | None` to `ConsoleChatMessage`.
- **Modify** `tldw_chatbook/Chat/console_chat_store.py` — in-memory tree (nodes + children + active-leaf), real `parent_message_id` write-through, sibling API, active-path view, active-leaf accessors, sync-sequence rework, regenerate-as-sibling primitive.
- **Modify** `tldw_chatbook/Chat/console_chat_controller.py` — `regenerate_message` creates a sibling node and streams into it (no variant stream).
- **Modify** `tldw_chatbook/Chat/console_message_actions.py` — `<` / `>` gate on "has siblings" (via a new `sibling_count`/`sibling_index` read model) instead of `message.variants is not None`.
- **Modify** `tldw_chatbook/Widgets/Console/console_transcript.py` — render an `n/m` sibling counter; body renders the active node's content.
- **Modify** `tldw_chatbook/UI/Screens/chat_screen.py` — resume active-path reconstruction (carry `parent_message_id`, walk active-leaf ancestry, fallback + pointer repair); repoint `_select_console_message_variant` at sibling navigation.

Two design notes that shape every task:

1. **Sibling identity = separate message nodes.** Phase A drops the "N variants on one message" model in favor of "N sibling message nodes, one active." `ConsoleVariantSet` is *not* used to represent regenerations anymore; the transcript shows the active sibling and the action row exposes `<`/`>` when the active message's parent has more than one child. (The `ConsoleVariantSet` dataclass may remain in the file, unused by regenerate, until Phase B/cleanup — do not delete it in Phase A to avoid churn in unrelated sync-metadata code.)
2. **Active path = `_messages_by_session[session_id]`.** Keep this list as the materialized active path so the ~dozen existing readers (`messages_for_session`, `_provider_messages_for_session`, persistence, sync) keep working unchanged. A new `_nodes_by_session` + `_children_by_parent` + `_active_leaf_by_session` hold the full tree and off-path branches.

## Invariants (hold after every task)

These are the rules the tree refactor must not violate. Several tasks below reference them.

- **Full tree vs. view.** The full tree — **all** branches, on- and off-path — lives in
  `_nodes_by_session` (native id → live `ConsoleChatMessage` object), `_children_by_parent`
  (session → {native parent id | `None` → ordered child native ids}), and
  `_native_parent_by_message` (native id → native parent id | `None`). `_messages_by_session` is a
  **derived view** = the current active path only.
- **Single writer for the view.** `_messages_by_session[sid]` is written **only** by
  `_recompute_active_path(sid)`, which walks `_active_leaf_by_session[sid]` up via
  `_native_parent_by_message` to the root, reverses, and materializes the list from the **live node
  objects in `_nodes_by_session`** (never copies — streaming mutates content in place and the view
  must see it). Every mutation (append, create_sibling, set_active_leaf, delete) ends by calling it.
- **Id lookups use the full tree.** `_message_or_raise`, `get_message`, `siblings_at`,
  `set_active_leaf`, `_leaf_under`, etc. resolve nodes from `_nodes_by_session` /
  `_message_session_index` — **never** by scanning `_messages_by_session` (which omits off-path
  nodes). This is a direct change to `_message_or_raise` (currently scans the active-path list).
- **Every node is registered everywhere.** `append_message`, `create_sibling`, and
  `restore_persisted_session` each register a node in **all** of `_nodes_by_session`,
  `_children_by_parent`, `_native_parent_by_message`, and `_message_session_index`. `close_session`
  purges from all of them (iterate `_nodes_by_session`, not the view).
- **`parent_message_id` (message field) is the persisted parent id; native parent lives in
  `_native_parent_by_message`.** They differ pre-persist (`parent_message_id` is `None` until the
  parent has a persisted id) and are equal-in-meaning post-persist.
- **TOOL markers are display-only, never tree nodes.** Live agent markers arrive via
  `append_message(role=TOOL, persist=False)` (`ConsoleAgentBridge._append_marker`). A TOOL row must
  **not** register in `_nodes_by_session`/`_children_by_parent`/`_native_parent_by_message`, must
  **not** become the active leaf, and must **not** be parented — otherwise the next real message
  would parent at a marker and corrupt the chain (this would break even linear agent chats). It is
  appended to the active-path view for display only. Consequence: `_recompute_active_path` rebuilds
  the view from real tree nodes and therefore drops TOOL markers on a swipe — an accepted Phase A
  limitation (agent markers are re-derived correctly on resume today and fully reworked in Phase C;
  swipe is blocked during a live run, so this only affects re-viewing a completed agent run's
  markers after swiping, a niche case).

---

### Task 1: ChaChaNotes v22→v23 migration + active-leaf accessors

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (`_CURRENT_SCHEMA_VERSION` at :154; `migration_steps` dict ~:3935-3954; add method + constant near the other `_MIGRATE_*` blocks ~:2407 and `_migrate_from_*` methods ~:3777; add accessors near other conversation methods)
- Create: `tldw_chatbook/DB/migrations/chachanotes_v22_to_v23_conversation_active_leaf.sql`
- Test: `Tests/DB/test_chachanotes_active_leaf_migration.py`

**Interfaces:**
- Produces: `CharactersRAGDB.set_conversation_active_leaf(conversation_id: str, message_id: str | None) -> None`; `CharactersRAGDB.get_conversation_active_leaf(conversation_id: str) -> str | None`; a `conversations.active_leaf_message_id TEXT` column present at schema v23.

- [ ] **Step 1: Write the failing test**

```python
# Tests/DB/test_chachanotes_active_leaf_migration.py
import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _db(tmp_path):
    return CharactersRAGDB(str(tmp_path / "c.db"), client_id="test-client")


def test_fresh_db_is_v23_with_active_leaf_column(tmp_path):
    db = _db(tmp_path)
    with db.get_connection() as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = 'rag_char_chat_schema'"
        ).fetchone()["version"]
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    assert version == 23
    assert "active_leaf_message_id" in cols


def test_active_leaf_roundtrip_and_default_null(tmp_path):
    db = _db(tmp_path)
    conv_id = db.add_conversation({"title": "t", "character_id": None})
    assert db.get_conversation_active_leaf(conv_id) is None
    db.set_conversation_active_leaf(conv_id, "msg-123")
    assert db.get_conversation_active_leaf(conv_id) == "msg-123"
    db.set_conversation_active_leaf(conv_id, None)
    assert db.get_conversation_active_leaf(conv_id) is None


def test_active_leaf_write_does_not_bump_version_or_emit_sync(tmp_path):
    db = _db(tmp_path)
    conv_id = db.add_conversation({"title": "t", "character_id": None})
    with db.get_connection() as conn:
        v_before = conn.execute(
            "SELECT version FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()["version"]
        sync_before = conn.execute(
            "SELECT COUNT(*) AS n FROM sync_log WHERE entity_id = ?", (conv_id,)
        ).fetchone()["n"]
    db.set_conversation_active_leaf(conv_id, "msg-abc")
    with db.get_connection() as conn:
        v_after = conn.execute(
            "SELECT version FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()["version"]
        sync_after = conn.execute(
            "SELECT COUNT(*) AS n FROM sync_log WHERE entity_id = ?", (conv_id,)
        ).fetchone()["n"]
    assert v_after == v_before, "active-leaf write must not bump version"
    assert sync_after == sync_before, "active-leaf write must not emit a sync_log row"
```

> Note: the real API is `CharactersRAGDB.add_conversation(conv_data: dict) -> str | None` (`ChaChaNotes_DB.py:5492`). Read its docstring for required keys (it fills `id`/`root_id`/`client_id` defaults; it typically needs at least `title` and may accept `character_id=None`). Adjust the `add_conversation(...)` calls to match; the assertions are the contract. If `add_conversation` is awkward for a bare column test, insert a minimal `conversations` row directly via `db.get_connection()` instead.

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest Tests/DB/test_chachanotes_active_leaf_migration.py -v`
Expected: FAIL — `test_fresh_db_is_v23...` gets `version == 22` and no `active_leaf_message_id` column; the accessor tests error with `AttributeError`.

- [ ] **Step 3: Bump the schema version** (`ChaChaNotes_DB.py:154`)

```python
    _CURRENT_SCHEMA_VERSION = 23  # Adds conversations.active_leaf_message_id (Console branching Phase A).
```

- [ ] **Step 4: Add the migration SQL constant** (place beside the other `_MIGRATE_*_SQL` constants, after `_MIGRATE_V21_TO_V22_SQL`)

```python
    # Keep this runner SQL aligned with
    # tldw_chatbook/DB/migrations/chachanotes_v22_to_v23_conversation_active_leaf.sql.
    # NOTE: no trigger DDL. `active_leaf_message_id` is a LOCAL-ONLY pointer that
    # must never reach sync_log, so the conversations_sync_* triggers are left
    # untouched and the column is never added to their payloads.
    _MIGRATE_V22_TO_V23_SQL = """
UPDATE db_schema_version
   SET version = 23
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 22;
"""
```

- [ ] **Step 5: Add the guarded migration method** (beside `_migrate_from_v21_to_v22`)

```python
    def _migrate_from_v22_to_v23(self, conn: sqlite3.Connection):
        """Migrate schema V22→V23: add the local-only ``active_leaf_message_id``
        pointer column to ``conversations``. No triggers change — the column is
        never synced (see ``set_conversation_active_leaf``)."""
        logger.info(f"Migrating schema from V22 to V23 for '{self._SCHEMA_NAME}' in DB: {self.db_path_str}...")
        try:
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "active_leaf_message_id" not in existing_columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN active_leaf_message_id TEXT")
            conn.executescript(self._MIGRATE_V22_TO_V23_SQL)
            logger.debug(f"[{self._SCHEMA_NAME} V22→V23] Migration script executed.")
            final_version = self._get_db_version(conn)
            if final_version != 23:
                raise SchemaError(
                    f"[{self._SCHEMA_NAME} V22→V23] Migration version check failed. Expected 23, got: {final_version}"
                )
            logger.info(f"[{self._SCHEMA_NAME} V22→V23] Migration completed successfully for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logger.opt(exception=True).error(f"[{self._SCHEMA_NAME} V22→V23] Migration failed: {e}")
            raise SchemaError(f"Migration from V22 to V23 failed for '{self._SCHEMA_NAME}': {e}") from e
        except Exception as e:
            logger.opt(exception=True).error(f"[{self._SCHEMA_NAME} V22→V23] Unexpected error during migration: {e}")
            raise SchemaError(f"Unexpected error migrating from V22 to V23 for '{self._SCHEMA_NAME}': {e}") from e
```

- [ ] **Step 6: Register the migration** in the `migration_steps` dict (after the `21:` line, ~:3953)

```python
                    21: self._migrate_from_v21_to_v22,
                    22: self._migrate_from_v22_to_v23,
```

- [ ] **Step 7: Add the accessors** (near the other conversation read/update methods)

```python
    def set_conversation_active_leaf(
        self, conversation_id: str, message_id: str | None
    ) -> None:
        """Set the local-only active-leaf pointer for a conversation.

        Deliberately a bare UPDATE that does NOT bump ``version``/``last_modified``
        and touches no column named in the ``conversations_sync_update`` trigger
        WHEN clause, so it never emits a ``sync_log`` row. Last-write-wins; no
        optimistic locking (this is a per-client view pointer, not synced state).
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE conversations SET active_leaf_message_id = ? "
                "WHERE id = ? AND deleted = 0",
                (message_id, conversation_id),
            )

    def get_conversation_active_leaf(self, conversation_id: str) -> str | None:
        """Return the local-only active-leaf pointer, or ``None`` if unset/missing."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT active_leaf_message_id FROM conversations "
                "WHERE id = ? AND deleted = 0",
                (conversation_id,),
            ).fetchone()
        return row["active_leaf_message_id"] if row else None
```

> Confirm `self.transaction()` / `self.get_connection()` are the correct context managers used elsewhere in this file (they are — see `update_conversation`). Use the same row-access style (`row["col"]`) the file uses.

- [ ] **Step 8: Create the `.sql` reference file** `tldw_chatbook/DB/migrations/chachanotes_v22_to_v23_conversation_active_leaf.sql`

```sql
-- Migration: ChaChaNotes V22 to V23 — conversations.active_leaf_message_id
-- Adds a nullable, LOCAL-ONLY pointer column recording which leaf message is
-- the active branch tip for the native Console. It is intentionally NOT synced:
-- the setter writes it without bumping version/last_modified, so the
-- conversations_sync_* triggers never fire and the column is never in a sync
-- payload. The runner guards the ADD COLUMN with a PRAGMA check (SQLite has no
-- ADD COLUMN IF NOT EXISTS) so replayed/partial migrations are idempotent.

ALTER TABLE conversations ADD COLUMN active_leaf_message_id TEXT;

UPDATE db_schema_version
   SET version = 23
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 22;
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest Tests/DB/test_chachanotes_active_leaf_migration.py -v`
Expected: PASS (3 tests). If `add_conversation` needed different args, fix the test setup, not the assertions.

- [ ] **Step 10: Commit**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/DB/migrations/chachanotes_v22_to_v23_conversation_active_leaf.sql Tests/DB/test_chachanotes_active_leaf_migration.py
git commit -m "feat(db): add local-only conversations.active_leaf_message_id (v22->v23)"
```

---

### Task 2: `parent_message_id` on `ConsoleChatMessage`

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py:197-212`
- Test: `Tests/Chat/test_console_chat_models.py` (add to existing, or create)

**Interfaces:**
- Produces: `ConsoleChatMessage.parent_message_id: str | None = None` (the persisted parent id of this node's parent, i.e. the *persisted* id, mirroring `persisted_message_id`); plus transient render fields `sibling_index: int = 0` and `sibling_count: int = 1` (populated by the store on active-path snapshots; not persisted).

- [ ] **Step 1: Write the failing test**

```python
def test_console_chat_message_has_parent_message_id_default_none():
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
    msg = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi")
    assert msg.parent_message_id is None
    msg2 = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="yo", parent_message_id="p1")
    assert msg2.parent_message_id == "p1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_models.py -k parent_message_id -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'parent_message_id'`.

- [ ] **Step 3: Add the field** (`console_chat_models.py`, after `persisted_message_id`)

```python
    persisted_message_id: str | None = None
    #: Persisted id of this node's PARENT in the conversation tree (None for a
    #: root / not-yet-known parent). Distinct from ``persisted_message_id``
    #: (this node's own persisted id). Used to reconstruct the active path.
    parent_message_id: str | None = None
    #: Transient (non-persisted) sibling-navigation hints the store fills in on
    #: active-path snapshots so the renderer can show `<`/`>` + an `n/m` counter
    #: without reaching into store internals. Default 0/1 = "no siblings".
    sibling_index: int = 0
    sibling_count: int = 1
```

Update the Step 1 test to also assert the sibling defaults:

```python
    assert (msg.sibling_index, msg.sibling_count) == (0, 1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_models.py -k parent_message_id -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py Tests/Chat/test_console_chat_models.py
git commit -m "feat(console): add parent_message_id to ConsoleChatMessage"
```

---

### Task 3: Store — in-memory tree scaffolding + active-leaf accessors

Introduce the tree structures and pointer accessors **without changing send/regenerate behavior yet**. After this task the store still behaves linearly (active path == full transcript), but the plumbing exists.

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (`__init__` :226-241; `create_session` :257-273; `restore_persisted_session` :275-309; `restore_state` :557-594; `close_session` :377-410; `append_message` :619-656)
- Test: `Tests/Chat/test_console_chat_store_tree.py`

**Interfaces:**
- Produces on `ConsoleChatStore`:
  - `_nodes_by_session: dict[str, dict[str, ConsoleChatMessage]]` — session → {native id → node}
  - `_children_by_parent: dict[str, dict[str | None, list[str]]]` — session → {parent native id (or None for roots) → ordered child native ids}
  - `_active_leaf_by_session: dict[str, str | None]` — session → active-leaf native id
  - `active_leaf(session_id: str) -> str | None`
  - `set_active_leaf(session_id: str, message_id: str | None) -> None` (updates in-memory pointer, recomputes the active-path view into `_messages_by_session`, and write-through persists to `conversations.active_leaf_message_id` via `getattr(self.persistence, "db", None)` when a persisted conversation exists)
  - `siblings_at(message_id: str) -> tuple[list[ConsoleChatMessage], int, int]` — (ordered sibling snapshots, index of `message_id`, count)
  - `active_path_message_ids(session_id: str) -> list[str]` — native ids from root→active leaf

- [ ] **Step 1: Write failing tests**

```python
# Tests/Chat/test_console_chat_store_tree.py
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


def _store_with_session():
    store = ConsoleChatStore()
    session = store.create_session(title="t")
    store.active_session_id = session.id
    return store, session.id


def test_linear_append_tracks_tree_and_active_leaf():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="yo")
    # active leaf is the last appended node; active path is the full linear list
    assert store.active_leaf(sid) == a.id
    assert store.active_path_message_ids(sid) == [u.id, a.id]
    # each node's in-memory parent is the previous active leaf
    assert store.get_message(a.id).parent_message_id is None  # persisted parent unknown w/o persistence
    # a single child => no siblings
    sibs, idx, count = store.siblings_at(a.id)
    assert count == 1 and idx == 0


def test_set_active_leaf_recomputes_active_path():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="one")
    store.set_active_leaf(sid, u.id)
    assert store.active_path_message_ids(sid) == [u.id]
    store.set_active_leaf(sid, a.id)
    assert store.active_path_message_ids(sid) == [u.id, a.id]
```

- [ ] **Step 2: Run to verify they fail**

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_store_tree.py -v`
Expected: FAIL — `AttributeError` on `active_leaf` / `active_path_message_ids` / `siblings_at`.

- [ ] **Step 3: Add the structures + maintain them**

Implementation notes (follow existing idioms + the **Invariants** section above; exact bodies are the implementer's, tests are the contract):

1. In `__init__` add the four dicts (empty): `_nodes_by_session`, `_children_by_parent`, `_native_parent_by_message`, `_active_leaf_by_session`.
2. In `create_session`, initialize `_nodes_by_session[id] = {}`, `_children_by_parent[id] = {}`, `_active_leaf_by_session[id] = None` alongside `_messages_by_session[id] = []`.
3. In `append_message`:
   - **If `role == TOOL`** (display-only marker; see Invariants): register in `_message_session_index` only, append to `_messages_by_session[sid]` for display, and return — do **not** touch tree structures or the active leaf.
   - **Otherwise** (real message): capture `old_leaf = _active_leaf_by_session[sid]`; register the node in `_nodes_by_session[sid]`; set `_native_parent_by_message[message.id] = old_leaf`; append its native id to `_children_by_parent[sid].setdefault(old_leaf, [])`; set `_active_leaf_by_session[sid] = message.id`; then call `_recompute_active_path(sid)` (the single writer of `_messages_by_session`). Leave `message.parent_message_id` as `None` here — it is set from the parent's *persisted* id at persist time (Task 4).
4. `_recompute_active_path(session_id)`: walk `_active_leaf_by_session[sid]` up via `_native_parent_by_message` to the root, reverse, and rebuild `_messages_by_session[sid]` **from the live node objects in `_nodes_by_session` (not copies)**; also fill each visited node's transient `sibling_index`/`sibling_count` from `_children_by_parent[sid][<that node's native parent>]`. (TOOL markers, being display-only, are re-appended by the marker system, not by recompute — accepted Phase A limitation.)
5. `active_leaf`, `set_active_leaf`, `active_path_message_ids`, `siblings_at`, `_leaf_under(node_id)` (walk `_children_by_parent` picking the last child at each step). `set_active_leaf` sets the pointer, calls `_recompute_active_path`, and — when a persisted conversation id and a `db` seam exist (mirror `persistence_db = getattr(self.persistence, "db", None)` from `persist_session_if_needed`) — write-throughs `db.set_conversation_active_leaf(conversation_id, <persisted id of the leaf, or None>)`.
6. **Change `_message_or_raise`** to resolve from `_nodes_by_session[session_id][message_id]` (per Invariants) so off-path nodes are findable; keep the `_message_session_index` guard for the session lookup.
7. In `restore_state` and `close_session`, clear/rebuild the four new dicts alongside the existing ones (iterate `_nodes_by_session` to purge in `close_session`), and fix the pre-existing `_variant_stream_bases` non-clear in `restore_state` while here.
8. **Update `delete_message`** (`console_chat_store.py:709-726`): today it only filters `_messages_by_session` + pops `_message_session_index`. Under the tree it must also remove the node (and its subtree) from `_nodes_by_session`/`_children_by_parent`/`_native_parent_by_message`, and — if the deleted node was on the active path — move `_active_leaf_by_session[sid]` to the deleted node's **parent** (then `_recompute_active_path`). Add a test: deleting the active leaf leaves the parent as the new leaf; deleting an off-path node doesn't change the active path. Keep the existing "can't delete pending/streaming" guard.

Add a test asserting a TOOL marker does **not** change the active leaf or become a parent:

```python
def test_tool_marker_is_display_only():
    st, sid = _s()  # helper from this test module
    u = st.append_message(sid, role=ConsoleMessageRole.USER, content="q")
    a = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a")
    st.append_message(sid, role=ConsoleMessageRole.TOOL, content="⚙ tool → ok")
    assert st.active_leaf(sid) == a.id            # marker did NOT become the leaf
    nxt = st.append_message(sid, role=ConsoleMessageRole.USER, content="q2")
    # q2 parents at the assistant answer, NOT the marker
    assert st._native_parent_by_message[nxt.id] == a.id
```

- [ ] **Step 4: Run to verify they pass**

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_store_tree.py -v`
Expected: PASS. Then run the existing store suite to confirm no regression:
Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_store.py -v`
Expected: PASS (linear behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store_tree.py
git commit -m "feat(console): in-memory conversation tree + active-leaf accessors (linear-equivalent)"
```

---

### Task 4: Store — write a real `parent_message_id` through to persistence

Today `_persist_new_message` hardcodes `parent_message_id=None` (`console_chat_store.py:1325`). Make it pass the persisted id of the node's in-memory parent, and set the message's own `parent_message_id` field from it. Also write the active-leaf pointer through on persist.

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (`_persist_new_message` :1312-1359)
- Test: `Tests/Chat/test_console_chat_store_parent_persist.py`

**Interfaces:**
- Consumes: Task 3 tree structures; `ConsoleChatPersistence.create_message(parent_message_id=...)` (already accepts it — `console_chat_store.py:71`).
- Produces: persisted messages carry a real `parent_message_id` chain for linear conversations (each message parented at the previous persisted message).

- [ ] **Step 1: Write the failing test** — use a fake persistence that records `parent_message_id`:

```python
# Tests/Chat/test_console_chat_store_parent_persist.py
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


class _RecordingPersistence:
    db = None
    def __init__(self): self.created = []
    def create_conversation(self, **kw): return "conv-1"
    def create_message(self, *, conversation_id, sender, content, image_data,
                       image_mime_type, message_id=None, parent_message_id=None,
                       feedback=None, attachments=None):
        pid = f"p{len(self.created)+1}"
        self.created.append({"id": pid, "parent_message_id": parent_message_id,
                             "content": content, "sender": sender})
        return pid
    def update_message_content(self, **kw): return True


def test_linear_persist_sets_parent_chain():
    p = _RecordingPersistence()
    store = ConsoleChatStore(persistence=p)
    s = store.create_session(title="t"); store.active_session_id = s.id
    store.append_message(s.id, role=ConsoleMessageRole.USER, content="hi", persist=True)
    store.append_message(s.id, role=ConsoleMessageRole.ASSISTANT, content="yo", persist=True)
    assert p.created[0]["parent_message_id"] is None          # first message: root
    assert p.created[1]["parent_message_id"] == p.created[0]["id"]  # second parented at first
```

- [ ] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_store_parent_persist.py -v`
Expected: FAIL — `p.created[1]["parent_message_id"]` is `None` (hardcoded).

- [ ] **Step 3: Thread the real parent** in `_persist_new_message` — replace the hardcoded `parent_message_id=None` in `create_kwargs` with the persisted id of this node's in-memory parent, and set `message.parent_message_id` to it before/after create:

```python
        parent_native_id = self._native_parent_by_message.get(message.id)
        parent_persisted_id = None
        if parent_native_id is not None:
            parent_node = self._nodes_by_session.get(session_id, {}).get(parent_native_id)
            parent_persisted_id = parent_node.persisted_message_id if parent_node else None
        message.parent_message_id = parent_persisted_id
        create_kwargs: dict[str, Any] = dict(
            conversation_id=conversation_id,
            sender=message.role.value,
            content=message.content,
            message_id=None,
            parent_message_id=parent_persisted_id,
            feedback=message.feedback,
        )
```

> Because persistence can be deferred (a parent may persist after a child is first created), verify the common ordering: in the linear flow the parent (user msg) persists before the child (assistant) — the user echo is appended+persisted first, then the assistant node. Add an assertion in the test above proving the ordering holds; if a child could persist before its parent, fall back to `None` (root) rather than a dangling id.

- [ ] **Step 4: Run to verify it passes** and re-run the full store suite.

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_store_parent_persist.py Tests/Chat/test_console_chat_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store_parent_persist.py
git commit -m "feat(console): persist real parent_message_id for the linear chain"
```

---

### Task 5: Store — `create_sibling` + regenerate-as-sibling primitive

Add the primitive that regenerate will use, and rework sync sequencing to be tree-aware (parent + sequence from the tree, not the flat list).

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (add `create_sibling`; rework `_previous_persisted_message_id` :1472-1482 and `_sync_message_sequence` :1452-1462 to walk the active path/tree)
- Test: `Tests/Chat/test_console_chat_store_sibling.py`

**Interfaces:**
- Produces: `create_sibling(anchor_message_id: str, *, role: ConsoleMessageRole, content: str = "", persist: bool = False) -> ConsoleChatMessage` — creates a new node whose **native parent == the anchor's native parent** (i.e. a sibling of the anchor), registers it in the tree, sets it as the active leaf (so the active path now ends at the new node), recomputes the active-path view, and (when `persist`) write-throughs with the correct `parent_message_id`.

- [ ] **Step 1: Write failing tests**

```python
# Tests/Chat/test_console_chat_store_sibling.py
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


def _s():
    st = ConsoleChatStore(); ses = st.create_session(title="t"); st.active_session_id = ses.id
    return st, ses.id


def test_create_sibling_of_last_assistant_makes_two_children():
    st, sid = _s()
    u = st.append_message(sid, role=ConsoleMessageRole.USER, content="q")
    a = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="ans-1")
    a2 = st.create_sibling(a.id, role=ConsoleMessageRole.ASSISTANT, content="ans-2")
    sibs, idx, count = st.siblings_at(a2.id)
    assert count == 2 and idx == 1
    assert st.active_leaf(sid) == a2.id
    assert st.active_path_message_ids(sid) == [u.id, a2.id]  # tail is the new sibling


def test_create_sibling_midconversation_truncates_visible_tail():
    st, sid = _s()
    u1 = st.append_message(sid, role=ConsoleMessageRole.USER, content="q1")
    a1 = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a1")
    u2 = st.append_message(sid, role=ConsoleMessageRole.USER, content="q2")
    a2 = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a2")
    # regenerate a1 (mid-conversation) -> new sibling under u1, tail (u2,a2) drops off-path
    a1b = st.create_sibling(a1.id, role=ConsoleMessageRole.ASSISTANT, content="a1-alt")
    assert st.active_path_message_ids(sid) == [u1.id, a1b.id]
    # swiping back to a1 restores the old tail
    st.set_active_leaf(sid, a2.id)
    assert st.active_path_message_ids(sid) == [u1.id, a1.id, u2.id, a2.id]
```

- [ ] **Step 2: Run to verify they fail**

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_store_sibling.py -v`
Expected: FAIL — no `create_sibling`.

- [ ] **Step 3: Implement `create_sibling`** — build a `ConsoleChatMessage` like `append_message` does, but set its native parent to `self._native_parent_by_message[anchor_message_id]` (the anchor's parent), register in `_nodes_by_session` + `_children_by_parent[<that parent>]`, set `_native_parent_by_message[new.id]`, then `set_active_leaf(session_id, new.id)` (which recomputes the active path). When `persist`, call the deferred/persist path. Reuse `_recompute_active_path`.

- [ ] **Step 4: Rework sync sequencing to tree-aware** — `_previous_persisted_message_id` should return the persisted id of the node's **tree parent** (native-parent lookup → persisted id), and `_sync_message_sequence` should count sync-eligible messages **along the active path** (iterate `_messages_by_session[sid]`, which is now the active path) rather than assuming a single linear history. The existing bodies already iterate `_messages_by_session`, so once that list is the active path they are close to correct; adjust `_previous_persisted_message_id` to use the native-parent map instead of "previous in list".

- [ ] **Step 5: Run to verify they pass** + full store suite.

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_chat_store_sibling.py Tests/Chat/test_console_chat_store.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store_sibling.py
git commit -m "feat(console): create_sibling primitive + tree-aware sync sequencing"
```

---

### Task 6: Controller — regenerate creates a sibling node and streams into it

Replace the variant-stream regenerate with sibling creation. `regenerate_message` currently streams into the *same* message with `variant_mode=True` (`console_chat_controller.py:1252-1259`); change it to create a sibling assistant node under the anchor's parent and stream into the **new** node normally.

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`regenerate_message` :1201-1259; verify `_provider_messages_for_session` :2715-2728 and `_stream_assistant_response` behavior with `variant_mode=False`)
- Test: `Tests/Chat/test_console_regenerate_branching.py`

**Interfaces:**
- Consumes: `store.create_sibling(...)`, `store.set_active_leaf(...)`, existing `_stream_assistant_response(assistant_message_id=<new id>, variant_mode=False, ...)`.

- [ ] **Step 1: Write the failing test** (controller-level with a fake provider gateway/stream, following the pattern in the existing controller tests — locate `Tests/Chat/test_console_chat_controller*.py` and reuse its harness/fakes):

```python
# assert-level contract (adapt to the existing controller test harness):
# after regenerate_message(a1.id):
#   - store has TWO assistant children under a1's parent
#   - active leaf is the NEW child, not a1
#   - the new child's content is the freshly streamed text
#   - a1 (and any old tail under it) still exist off the active path
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Rewrite `regenerate_message`** — after validation + provider resolution, do:

```python
        new_message = self.store.create_sibling(
            message_id, role=ConsoleMessageRole.ASSISTANT, content="",
        )
        provider_messages = self._provider_messages_for_session(
            session_id, before_message_id=new_message.id,
        )
        # ... existing _ensure_user_continuation_instruction / _has_user_turn /
        #     skill / dictionaries / world-info / prefill steps, unchanged ...
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=new_message.id,
            variant_mode=False,          # normal stream into the new node
            prefill=prefill,
        )
```

> `_provider_messages_for_session(before_message_id=new_message.id)` yields the active path up to (excluding) the new node — i.e. the anchor's ancestry — because `create_sibling` made the new node the active-path tail. Confirm `create_sibling` inserts the new node into `_messages_by_session` before this call (it does, via `set_active_leaf`→recompute).
> On stream **failure**, the new node becomes a `failed` node on the active path (retryable) rather than restoring the old sibling — this is the intended node-model behavior; note it in the controller docstring.

- [ ] **Step 4: Run to verify it passes**, plus the existing controller + store suites.

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_regenerate_branching.py Tests/Chat/ -k "controller or regenerate" -v`
Expected: PASS. Investigate any variant-related controller test that assumed in-place regeneration; update it to the sibling contract.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_regenerate_branching.py
git commit -m "feat(console): regenerate creates a persisted sibling branch"
```

---

### Task 7: Swipe + counter — navigate siblings, show `n/m`

Make `<` / `>` move the active leaf across siblings, gate the buttons on "active message has siblings," and render an `n/m` counter.

**Files:**
- Modify: `tldw_chatbook/Chat/console_message_actions.py` (`available_actions` :101-132 — gate `_VARIANT_NAV_ACTIONS` on a sibling read model; `_variant_action_enabled` :282-287)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_select_console_message_variant` :12565-12577 → move active leaf; keep action ids `variant-previous`/`variant-next`)
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`_message_render_text` :170-197 — append ` (n/m)` when the message has siblings)
- Test: `Tests/Chat/test_console_sibling_nav.py`, `Tests/UI/test_console_native_transcript.py` (counter)

**Interfaces:**
- Consumes: `store.siblings_at(message_id) -> (list, index, count)`, `store.set_active_leaf`.
- Produces: `<`/`>` shown iff `count > 1`; navigation moves the active leaf to the previous/next sibling's most-recent-descendant leaf.

- [ ] **Step 1: Write failing tests**

```python
def test_sibling_nav_moves_active_leaf(...):
    # given two assistant siblings a1,a2 (a2 active), pressing variant-previous
    # sets active leaf back onto a1's subtree; variant-next returns to a2.
def test_counter_rendered_for_siblings(...):
    # a message with 2 siblings renders "(2/2)" (or chosen format) in its row.
def test_no_counter_for_single_child(...):
    # a linear message renders no counter.
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement.**
- `_select_console_message_variant(message_id, direction)`: get `(sibs, idx, count)`; compute target sibling by `idx-1`/`idx+1` (clamp/no-op at ends); `store.set_active_leaf(sid, <target sibling's leaf native id>)`. The store should expose a helper to resolve a sibling's most-recent-descendant leaf (add `_leaf_under(node_id)` walking `children[-1]` in-memory). Then `await self._sync_native_console_chat_ui()`.
- **Sibling info reaches the renderer via the snapshot fields** `sibling_index`/`sibling_count` (Task 2), which `_recompute_active_path` fills in (Task 3). No store threading into the pure `ConsoleMessageActionService` and no store reference in the transcript needed.
- `console_message_actions.available_actions`: gate the `_VARIANT_NAV_ACTIONS` (`<`/`>`) on `message.sibling_count > 1` (replacing the `message.variants is not None` gate); `_variant_action_enabled` returns `sibling_index > 0` for `variant-previous` and `sibling_index < sibling_count - 1` for `variant-next`.
- Counter: in the transcript row builder (`_message_render_text` :170-197), when `message.sibling_count > 1`, append `f" ({message.sibling_index + 1}/{message.sibling_count})"` per the existing label style.
- `_select_console_message_variant`: use `store.siblings_at(message_id)` to find the target sibling by direction, then `store.set_active_leaf(sid, store._leaf_under(<target sibling native id>))`; then `await self._sync_native_console_chat_ui()`.

> `message.variants` is no longer populated by regenerate (Task 6), so the old `variants`-based gate is dead; the new gate is `sibling_count`-based. Leave the `ConsoleVariantSet` dataclass and its (now-unused) `_sync_variant_metadata` path in place for Phase A to avoid churn in the sync module.

- [ ] **Step 4: Run to verify they pass** + transcript suite.

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/test_console_sibling_nav.py Tests/UI/test_console_native_transcript.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_transcript.py Tests/Chat/test_console_sibling_nav.py Tests/UI/test_console_native_transcript.py
git commit -m "feat(console): <>/counter navigate persisted sibling branches"
```

---

### Task 8: Resume — load the full tree + reconstruct the active path from the pointer

**Critical:** resume must load the **entire tree (all branches)** into the store, not just the
active path — otherwise off-path siblings aren't in memory and swipe-after-resume silently breaks
(everything shows `1/1`). `get_conversation_tree` already returns every branch, so this is plumbing,
not extra queries. The store builds the full tree and derives the active-path *view* from the leaf
pointer.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_console_messages_from_conversation_tree` :3933-3981 → return **all** nodes; the `_iter_console_tree_messages` `children[-1]` walk :3899-3915 is no longer the resume path — keep it only as the fallback leaf-resolver; `_resume_console_workspace_conversation` :4122-4273 — read the pointer + raise `depth_cap`)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (`restore_persisted_session` :275-309 → ingest the full node set + build the tree + compute the active-path view)
- Test: `Tests/UI/test_console_resume_active_path.py` (real DB round-trip)

**Interfaces:**
- Consumes: `db.get_conversation_active_leaf(conversation_id)`; tree node dicts (each carries `id`, `parent_message_id`, `role`/`sender`, `content`, `timestamp` — Agent D confirmed).
- Produces: `restore_persisted_session(*, title, workspace_id, persisted_conversation_id, all_nodes: list[ConsoleChatMessage], active_leaf_persisted_id: str | None, settings=None) -> ConsoleChatSession` (signature change: `messages` → `all_nodes` + `active_leaf_persisted_id`). It builds the full in-memory tree from `all_nodes` and sets `_messages_by_session` to the active-path view.

- [ ] **Step 1: Write failing tests** (real in-memory DB round-trip). Build with `ConsoleChatStore(persistence=<real chat_persistence_service over CharactersRAGDB(":memory:")>)`:
  1. Persist U1→A1, regenerate A1 → A1' (two assistant siblings), set the active leaf to the **older** sibling A1.
  2. Tear down the store; resume the conversation through the resume path.
  3. Assert the restored transcript matches the **A1** branch (not `children[-1]`=A1').
  4. **Assert swipe-after-resume works:** `siblings_at(<restored A1 id>)` reports `count == 2`, and `set_active_leaf` to the other sibling swaps the visible transcript — proving the off-path sibling was loaded into memory.

- [ ] **Step 2: Run to verify they fail** (current resume returns A1' via `children[-1]` and never loads A1' as a navigable sibling).

- [ ] **Step 3: Implement.**
- `_console_messages_from_conversation_tree` → **flatten the *entire* tree** (walk every node, all children) into a `list[ConsoleChatMessage]`, each carrying `persisted_message_id=row["id"]` and `parent_message_id=row.get("parent_message_id")`. Do **not** collapse to one path here.
- `_resume_console_workspace_conversation`: read `active_leaf_id = getattr(db, "get_conversation_active_leaf", lambda _c: None)(target)` (db via `getattr(self.app_instance, "chachanotes_db", None)`); call `get_conversation_tree` with a **raised `depth_cap`** (e.g. `depth_cap=10_000`, `root_limit` high) via the scope-service seam so a long/branchy conversation isn't truncated; pass `all_nodes` + `active_leaf_id` into `restore_persisted_session`.
- Marker injection stays as today but now targets the **active-path view**: after `restore_persisted_session`, inject `_inject_resume_agent_markers(store.messages_for_session(session.id), target)` into the rendered transcript (display-only overlay; not stored as tree nodes — see the TOOL-marker invariant).
- `restore_persisted_session(all_nodes, active_leaf_persisted_id, ...)`:
  1. Give each node a fresh native id (default) and its `persisted_message_id`. Build `persisted_id → native_id`.
  2. Register every node in `_nodes_by_session`, `_message_session_index`.
  3. Set `_native_parent_by_message[native] = persisted_to_native.get(node.parent_message_id)` (`None` for roots / unknown parents); build `_children_by_parent` from those links (children ordered by the DB's timestamp order, which the tree already provides).
  4. Resolve the active leaf: `native = persisted_to_native.get(active_leaf_persisted_id)`; if `active_leaf_persisted_id` is `None`/missing/points at an unknown or `deleted` node, **fall back** to the `children[-1]` most-recent-child leaf and then `db.set_conversation_active_leaf(conversation_id, <that leaf's persisted id>)` to repair.
  5. Set `_active_leaf_by_session[session.id]` and call `_recompute_active_path(session.id)`.

> Handle multiple `root_threads` (Agent D: the tree can have >1 root): the active leaf's ancestry selects exactly one root; other roots stay off-path (loaded but not shown), consistent with the tree model.

- [ ] **Step 4: Run to verify it passes** + the console resume suite.

Run: `source .venv/bin/activate && python -m pytest Tests/UI/test_console_resume_active_path.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_chat_store.py Tests/UI/test_console_resume_active_path.py
git commit -m "feat(console): resume reconstructs the active branch via the leaf pointer"
```

---

### Task 9: End-to-end integration test + regression sweep

**Files:**
- Test: `Tests/integration/test_console_branching_e2e.py`
- No production changes expected (fix any surfaced).

- [ ] **Step 1: Write the E2E test** over a real in-memory DB + store + controller (reuse the controller test harness's fake provider):
  1. Send U1→A1 (linear); assert one active path.
  2. Regenerate A1 → A1'; assert two siblings, active leaf = A1', `1/2`↔`2/2` navigable.
  3. Continue on A1' → U2→A2.
  4. Persist, drop the store, resume the conversation; assert the transcript == `[U1, A1', U2, A2]` and `active_leaf` persisted/restored.
  5. Swipe A1'↔A1; assert the tail swaps (`[U1,A1']`/`[U1,A1,...]`) and survives a second resume.
  6. Assert `sync_log` has no rows attributable to active-leaf writes (only real message writes).

- [ ] **Step 2: Run it; fix any integration gaps.**

Run: `source .venv/bin/activate && python -m pytest Tests/integration/test_console_branching_e2e.py -v`

- [ ] **Step 3: Full regression sweep** of touched areas.

Run: `source .venv/bin/activate && python -m pytest Tests/Chat/ Tests/DB/test_chachanotes_active_leaf_migration.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_resume_active_path.py -q`
Expected: PASS. Triage any variant/regenerate tests that encoded the old in-place behavior and update them to the sibling contract (documented in the task's Implementation Notes).

- [ ] **Step 4: Commit**

```bash
git add Tests/integration/test_console_branching_e2e.py
git commit -m "test(console): end-to-end branching persistence + resume + swipe"
```

---

## Self-Review

**Spec coverage (Phase A rows of the spec):**
- v22→v23 local-only pointer column, no trigger redefinition → Task 1. ✅
- `ConsoleChatMessage.parent_message_id` → Task 2. ✅
- In-memory tree + active-path view + active-leaf accessors → Task 3. ✅
- Real `parent_message_id` write-through (net-new local linking) → Task 4. ✅
- `create_sibling` + tree-aware sync sequencing → Task 5. ✅
- Regenerate → persisted sibling; mid-conversation fork truncates tail off-path → Task 6 (+ store Task 5 tests). ✅
- Swipe navigates siblings + `n/m` counter, gated on `count>1` → Task 7. ✅
- Resume active-leaf ancestry + fallback + pointer repair → Task 8. ✅
- Deferred/`persistence is None` still works → exercised by Tasks 3/5 (no-persistence stores) and Task 4/8 (with persistence). ✅
- Deterministic sibling order (timestamp, id) → tree children come back `ORDER BY timestamp` from `_build_message_tree`; in-memory children keep insertion order = creation order. Add an explicit ordering assertion in Task 5 if creation order and timestamp order could diverge. ✅ (note)
- Out of Phase A: user-message "Edit & resend" (Phase B), agent-marker anchoring (Phase C), tree browser / explicit fork action. ✅ excluded.

**Placeholder scan:** the deep store/controller tasks (3, 5, 6, 8) intentionally give exact interfaces + test contracts + key code rather than a full pasted method body, because those bodies are large refactors of code quoted inline and the tests are the precise contract. This is a deliberate altitude choice, not a TBD — each step names the exact method, file:line, and the observable behavior. Tasks 1, 2, 4, 7 carry complete code.

**Type consistency:** `active_leaf(session_id) -> str | None`, `set_active_leaf(session_id, message_id)`, `siblings_at(message_id) -> (list, int, int)`, `_leaf_under(node_id) -> str`, `create_sibling(anchor_message_id, *, role, content="", persist=False) -> ConsoleChatMessage`, `get_conversation_active_leaf(conversation_id) -> str | None`, `set_conversation_active_leaf(conversation_id, message_id)`, and the changed `restore_persisted_session(*, title, workspace_id, persisted_conversation_id, all_nodes, active_leaf_persisted_id, settings=None)` — used consistently across Tasks 3/5/6/7/8. `parent_message_id` (message field) = **persisted** parent id; **native** parent links live in `_native_parent_by_message`/`_children_by_parent`; the full tree lives in `_nodes_by_session` and `_messages_by_session` is the derived active-path view (Invariants).

**Review corrections folded in (2026-07-22, post-plan review):** (1) resume loads the **full tree**, not just the active path, so siblings are navigable after resume (Task 8 rewritten; `restore_persisted_session` signature changed). (2) All id-lookups (`_message_or_raise` et al.) resolve from `_nodes_by_session`, not the active-path view (Task 3). (3) `_native_parent_by_message` elevated to a first-class structure + the Invariants section. (4) `_recompute_active_path` is the single writer of the view, over live node objects. (5) sibling counts flow via transient `sibling_index`/`sibling_count` snapshot fields (Tasks 2/3/7). (6) resume raises `depth_cap`. (7) `delete_message` rewritten for the tree (Task 3, note 8). (8) `add_conversation(dict)` API confirmed (Task 1). (9) **TOOL markers are display-only, never tree nodes** (Invariants + Task 3) — otherwise they corrupt the parent chain even in linear agent chats.

**Known limitations recorded in-plan:** ordinal agent-marker placement stays wrong for agent+branched convos, and TOOL markers drop from the view on swipe, until Phase C; failed regenerate leaves a retryable failed sibling rather than restoring the prior reply.
