# Console `/rewind` Before-First Restart Durability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the Console's deliberate “immediately before this first prompt” position so restart restores an empty active path and the selected prompt row's durable text without changing legacy unset, Sync, export, or branch behavior.

**Architecture:** Extend the local conversation cursor from one nullable leaf ID to two atomically written nullable IDs. `ConsoleChatStore` owns validation and in-memory state, while the database owns only the local tri-state representation; hydration maps both persisted IDs after rebuilding the complete tree. The first-prompt UI path uses a dedicated honest-result operation, and every ordinary durable leaf advance clears the companion marker through the existing scalar setter's delegated atomic write.

**Tech Stack:** Python 3.11+, SQLite migrations, Textual 8.x, pytest/pytest-asyncio, existing Console tree/store and Chat persistence services.

**ADR required:** yes — existing [ADR-098](../../../backlog/decisions/098-console-active-path-before-first-cursor.md) governs this plan.

**Approved spec:** [Console `/rewind` before-first restart durability](../specs/2026-08-28-console-rewind-before-first-restart-design.md)

---

## File and responsibility map

| File | Responsibility in this change |
| --- | --- |
| `tldw_chatbook/DB/migrations/chachanotes_v53_to_v54_active_leaf_before_message.sql` | Guarded v53→v54 version stamp for the local-only cursor companion. |
| `tldw_chatbook/DB/ChaChaNotes_DB.py` | Register v54 and expose atomic two-column cursor APIs while preserving scalar APIs. |
| `Tests/DB/test_chachanotes_v54_before_first_cursor.py` | Migration, cursor round-trip, missing-row, and local-only invariants. |
| `Tests/DB/test_chachanotes_v53_safe_capture_trim.py` | Keep v53 behavior tests valid when the current end-of-chain version becomes 54. |
| `Tests/DB/test_chachanotes_console_thinking_migration.py` | Keep the v52 behavior test pinned to the current end-of-chain version rather than literal 53. |
| `tldw_chatbook/Chat/console_chat_store.py` | Dedicated before-message mutation, validation, tri-state resume, draft hydration, and repair. |
| `Tests/Chat/test_console_chat_store_before_first.py` | Focused store mutation/resume state-machine tests. |
| `tldw_chatbook/Chat/console_conversation_hydration.py` | Read and pass both cursor components through production hydration. |
| `Tests/Chat/test_console_conversation_hydration.py` | Prove production hydration consumes the pair. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Route index-zero restore and warn honestly on failed durability. |
| `Tests/UI/test_console_rewind_restore.py` | Screen routing, composer refill, and warning behavior. |
| `tldw_chatbook/Chat/console_dispatch_repository.py` | Clear the marker in the transaction accepting a new leaf. |
| `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py` | Prove acceptance atomically clears stale before-first state. |
| `Tests/UI/test_console_resume_active_path.py` | Real-DB canonical/legacy resume and repair coverage. |
| `Tests/integration/test_console_rewind_e2e.py` | Persist/drop/resume/resend lifecycle and fixture correction. |
| `backlog/tasks/task-574 - Console-rewind-restore-to-before-first-message-not-restart-durable.md` | Completion evidence and status. |

No export, import, Sync payload, fork snapshot, trajectory format, attachment-restaging, or legacy-root-repair module changes are planned.

### Task 1: Add schema v54 and the atomic local cursor API

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v53_to_v54_active_leaf_before_message.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:575,6872-7031,7190-7233,10872-10898`
- Create: `Tests/DB/test_chachanotes_v54_before_first_cursor.py`
- Modify: `Tests/DB/test_chachanotes_v53_safe_capture_trim.py:183-417`
- Modify: `Tests/DB/test_chachanotes_console_thinking_migration.py:254-270`

- [ ] **Step 1: Write failing v53→v54 migration tests.**

Use `Tests.ChaChaNotesDB.historical_bootstrap.chachanotes_db_at_version` for a genuine v53 database. Cover both the ordinary upgrade and the interrupted shape where the new column already exists while the stamp remains 53.

```python
def test_v54_adds_local_before_message_cursor(tmp_path: Path) -> None:
    path = tmp_path / "v53.sqlite"
    with chachanotes_db_at_version(path, 53):
        pass
    upgraded = CharactersRAGDB(path, client_id="v54-upgrade")
    try:
        with upgraded.get_connection() as conn:
            columns = {row[1] for row in conn.execute(
                "PRAGMA table_info(conversations)"
            )}
            version = conn.execute(
                "SELECT version FROM db_schema_version "
                "WHERE schema_name = 'rag_char_chat_schema'"
            ).fetchone()[0]
        assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION == 54
        assert "active_leaf_before_message_id" in columns
    finally:
        upgraded.close_connection()


def test_v54_reenters_when_column_exists_but_stamp_is_v53(tmp_path: Path) -> None:
    path = tmp_path / "partial.sqlite"
    with chachanotes_db_at_version(path, 53) as db:
        with db.transaction() as cursor:
            cursor.execute(
                "ALTER TABLE conversations "
                "ADD COLUMN active_leaf_before_message_id TEXT"
            )
    recovered = CharactersRAGDB(path, client_id="v54-recover")
    try:
        assert recovered._get_db_version(recovered.get_connection()) == 54
    finally:
        recovered.close_connection()
```

- [ ] **Step 2: Run the migration tests to verify RED.**

```bash
../../.venv/bin/python -m pytest Tests/DB/test_chachanotes_v54_before_first_cursor.py -q
```

Expected: FAIL because v54 and the companion column do not exist.

- [ ] **Step 3: Implement the guarded migration.**

Create the SQL file:

```sql
-- ChaChaNotes v53 -> v54: local explicit-before-first Console cursor.
UPDATE db_schema_version
   SET version = 54
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 53;
```

In `CharactersRAGDB`, bump `_CURRENT_SCHEMA_VERSION` to 54, register `53: self._migrate_from_v53_to_v54`, and add:

```python
def _migrate_from_v53_to_v54(self, conn: sqlite3.Connection) -> None:
    """Add the local-only explicit-before-first Console cursor."""
    self._require_migration_entry_version(conn, 53, "V53→V54")
    migration_path = (
        Path(__file__).parent
        / "migrations"
        / "chachanotes_v53_to_v54_active_leaf_before_message.sql"
    )
    try:
        with self.transaction() as cursor:
            columns = {
                row[1]
                for row in cursor.execute(
                    "PRAGMA table_info(conversations)"
                ).fetchall()
            }
            if "active_leaf_before_message_id" not in columns:
                cursor.execute(
                    "ALTER TABLE conversations "
                    "ADD COLUMN active_leaf_before_message_id TEXT"
                )
            self._execute_migration_statements(
                cursor,
                migration_path.read_text(encoding="utf-8"),
                "V53→V54",
            )
        if self._get_db_version(conn) != 54:
            raise SchemaError("[rag_char_chat_schema V53→V54] version check failed")
    except (OSError, sqlite3.Error, CharactersRAGDBError, SchemaError) as exc:
        raise SchemaError(
            f"Migration from V53 to V54 failed for '{self._SCHEMA_NAME}': {exc}"
        ) from exc
```

Do not add trigger DDL: omission from the conversation triggers is the local-only boundary.

Update only the end-of-chain assertions in the two directly affected historical migration suites. A database opened through current `CharactersRAGDB` now lands on `_CURRENT_SCHEMA_VERSION == 54`; the tests still prove their v52→v53 transformation through the capture/thinking assertions. Preserve assertions that intentionally inspect a database stopped at v52. In the safe-capture suite replace current-open literals at lines 194, 297, 394, and 417 with `CharactersRAGDB._CURRENT_SCHEMA_VERSION`; rename the fresh-database test from `lands_on_v53` to `lands_on_current`. In the thinking migration suite remove the trailing literal `== 53` while retaining equality with `_CURRENT_SCHEMA_VERSION`.

- [ ] **Step 4: Run Step 2 and verify GREEN.**

Expected: both migration tests PASS.

- [ ] **Step 5: Write failing atomic cursor API tests.**

```python
def test_cursor_round_trip_and_scalar_compatibility(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "cursor.sqlite", client_id="cursor")
    conversation_id = db.add_conversation({"title": "Cursor"})
    assert db.get_conversation_active_cursor(conversation_id) == (None, None)
    assert db.set_conversation_active_cursor(
        conversation_id,
        active_leaf_message_id=None,
        before_message_id="root-user",
    ) is True
    assert db.get_conversation_active_cursor(conversation_id) == (
        None,
        "root-user",
    )
    assert db.set_conversation_active_leaf(conversation_id, "assistant") is None
    assert db.get_conversation_active_cursor(conversation_id) == (
        "assistant",
        None,
    )
    assert db.get_conversation_active_leaf(conversation_id) == "assistant"
    assert db.set_conversation_active_cursor(
        "missing",
        active_leaf_message_id=None,
        before_message_id="root-user",
    ) is False


def test_cursor_write_is_local_only(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "local.sqlite", client_id="cursor")
    conversation_id = db.add_conversation({"title": "Cursor"})
    with db.get_connection() as conn:
        before = conn.execute(
            "SELECT version, last_modified FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        sync_count = conn.execute(
            "SELECT COUNT(*) FROM sync_log WHERE entity_id = ?",
            (conversation_id,),
        ).fetchone()[0]
    assert db.set_conversation_active_cursor(
        conversation_id,
        active_leaf_message_id=None,
        before_message_id="root-user",
    ) is True
    with db.get_connection() as conn:
        after = conn.execute(
            "SELECT version, last_modified FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        assert tuple(after) == tuple(before)
        assert conn.execute(
            "SELECT COUNT(*) FROM sync_log WHERE entity_id = ?",
            (conversation_id,),
        ).fetchone()[0] == sync_count
```

- [ ] **Step 6: Run Step 2 and verify RED with missing cursor methods.**

- [ ] **Step 7: Implement the minimal two-column API and scalar delegation.**

```python
def set_conversation_active_cursor(
    self,
    conversation_id: str,
    *,
    active_leaf_message_id: str | None,
    before_message_id: str | None,
) -> bool:
    """Atomically set the local-only Console cursor components."""
    with self.transaction() as conn:
        updated = conn.execute(
            "UPDATE conversations "
            "SET active_leaf_message_id = ?, "
            "active_leaf_before_message_id = ? "
            "WHERE id = ? AND deleted = 0",
            (active_leaf_message_id, before_message_id, conversation_id),
        )
    return updated.rowcount == 1

def get_conversation_active_cursor(
    self, conversation_id: str
) -> tuple[str | None, str | None]:
    """Return local active-leaf and explicit-before-first IDs."""
    with self.get_connection() as conn:
        row = conn.execute(
            "SELECT active_leaf_message_id, active_leaf_before_message_id "
            "FROM conversations WHERE id = ? AND deleted = 0",
            (conversation_id,),
        ).fetchone()
    if row is None:
        return None, None
    return row["active_leaf_message_id"], row["active_leaf_before_message_id"]

def set_conversation_active_leaf(
    self, conversation_id: str, message_id: str | None
) -> None:
    self.set_conversation_active_cursor(
        conversation_id,
        active_leaf_message_id=message_id,
        before_message_id=None,
    )

def get_conversation_active_leaf(self, conversation_id: str) -> str | None:
    active_leaf, _before = self.get_conversation_active_cursor(conversation_id)
    return active_leaf
```

- [ ] **Step 8: Run database regressions and verify GREEN.**

```bash
../../.venv/bin/python -m pytest \
  Tests/DB/test_chachanotes_v54_before_first_cursor.py \
  Tests/DB/test_chachanotes_active_leaf_migration.py \
  Tests/DB/test_chachanotes_v53_safe_capture_trim.py \
  Tests/DB/test_chachanotes_console_thinking_migration.py -q
```

- [ ] **Step 9: Commit the schema/API unit.**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/DB/migrations/chachanotes_v53_to_v54_active_leaf_before_message.sql \
  Tests/DB/test_chachanotes_v54_before_first_cursor.py \
  Tests/DB/test_chachanotes_v53_safe_capture_trim.py \
  Tests/DB/test_chachanotes_console_thinking_migration.py
git commit -m "feat(console): add durable before-first cursor"
```

### Task 2: Add the honest store mutation for positioning before a prompt

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:9714-9758,14658-14705`
- Create: `Tests/Chat/test_console_chat_store_before_first.py`

- [ ] **Step 1: Write failing store-operation tests.**

Use this minimal DB double and repeat the complete persisted-session setup for each named precondition rather than adding production abstraction:

```python
class _CursorDB:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[str, str | None, str | None]] = []

    def set_conversation_active_cursor(
        self,
        conversation_id: str,
        *,
        active_leaf_message_id: str | None,
        before_message_id: str | None,
    ) -> bool:
        self.calls.append(
            (conversation_id, active_leaf_message_id, before_message_id)
        )
        return self.result


def test_temporary_before_first_is_success_without_durable_write() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    session = store.create_session(title="Temporary")
    root = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="prompt"
    )
    assert store.set_active_path_before(session.id, root.id) is True
    assert store.active_path_message_ids(session.id) == []
    assert db.calls == []


def test_persisted_before_first_writes_marker_and_keeps_empty_path() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    root = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="prompt",
        persisted_message_id="root",
    )
    session = store.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id="conversation",
        all_nodes=[root],
        active_leaf_persisted_id="root",
    )
    assert store.set_active_path_before(session.id, root.id) is True
    assert store.active_path_message_ids(session.id) == []
    assert db.calls[-1] == ("conversation", None, "root")
```

Add these equally explicit tests:

- persisted conversation + target without `persisted_message_id` returns `False`, performs no DB call, and still empties the in-memory path;
- writer returns false and writer raises: both return `False` after applying the rewind;
- persistence DB or writer unavailable returns `False` after applying the rewind;
- wrong role or a durable restored node with `message.parent_message_id is not None` raises `ValueError` without mutation;
- temporary `U1→A1→U2` rejects U2 even though its persisted-parent field is null, because its native parent is A1;
- unknown message raises `KeyError` without mutation.

- [ ] **Step 2: Run tests to verify RED.**

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_console_chat_store_before_first.py -q
```

Expected: FAIL because `set_active_path_before` is absent.

- [ ] **Step 3: Implement `set_active_path_before` with the honest result.**

```python
def set_active_path_before(self, session_id: str, message_id: str) -> bool:
    session = self._session_or_raise(session_id)
    node = self._nodes_by_session.get(session_id, {}).get(message_id)
    if node is None:
        raise KeyError(f"Unknown Console message: {message_id}")
    has_durable_ancestry = (
        session.persisted_conversation_id is not None
        and node.persisted_message_id is not None
    )
    parent_id = (
        node.parent_message_id
        if has_durable_ancestry
        else self._native_parent_by_message.get(message_id)
    )
    if node.role is not ConsoleMessageRole.USER or parent_id is not None:
        raise ValueError("Before-first target must be a root user message.")

    previous_leaf = self._active_leaf_by_session.get(session_id)
    self._active_leaf_by_session[session_id] = None
    self._recompute_active_path(session_id)
    self._bump_payload_revision(session_id)
    if previous_leaf is not None:
        self._bump_conversation_context_epoch(session_id)

    conversation_id = session.persisted_conversation_id
    if conversation_id is None:
        return True
    before_message_id = node.persisted_message_id
    if before_message_id is None:
        logger.bind(
            session_id=session_id,
            conversation_id=conversation_id,
        ).warning("Console before-first cursor target is not durable.")
        return False
    persistence_db = getattr(self.persistence, "db", None)
    writer = getattr(persistence_db, "set_conversation_active_cursor", None)
    if not callable(writer):
        logger.bind(
            session_id=session_id,
            conversation_id=conversation_id,
        ).warning("Console before-first cursor persistence is unavailable.")
        return False
    try:
        return bool(
            writer(
                conversation_id,
                active_leaf_message_id=None,
                before_message_id=before_message_id,
            )
        )
    except Exception:
        logger.bind(
            session_id=session_id,
            conversation_id=conversation_id,
        ).exception(
            "Failed to persist Console before-first cursor; "
            "the in-memory rewind remains applied."
        )
        return False
```

Do not change `set_active_leaf(..., None)` semantics. It remains generic unset; its scalar DB setter now clears both columns.

The ancestry split is load-bearing: durable restored nodes use the imported pre-repair `parent_message_id`, while temporary/native-only nodes use `_native_parent_by_message`. A persisted conversation whose target lacks a durable message ID also uses native-parent validation, then returns `False` after applying a valid root rewind because no restart cursor can be written.

- [ ] **Step 4: Run Step 2 and verify GREEN.**

- [ ] **Step 5: Run existing active-leaf/tree regressions.**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_chat_store_tree.py \
  Tests/UI/test_console_resume_active_path.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the store mutation unit.**

```bash
git add tldw_chatbook/Chat/console_chat_store.py \
  Tests/Chat/test_console_chat_store_before_first.py
git commit -m "feat(console): persist rewind before first prompt"
```

### Task 3: Restore the tri-state cursor through production hydration

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:1886-2025,14284-14355`
- Modify: `tldw_chatbook/Chat/console_conversation_hydration.py:425-490`
- Modify: `Tests/Chat/test_console_chat_store_before_first.py`
- Modify: `Tests/Chat/test_console_conversation_hydration.py`

- [ ] **Step 1: Write failing store resume state-machine tests.**

Add complete fixtures for:

1. `(None, None)` → newest fallback and repair.
2. `(valid leaf, None)` → selected branch.
3. `(None, valid root USER)` → empty path plus current durable content via `set_session_draft`.
4. `(valid leaf, marker)` → leaf wins and marker clears.
5. `(dangling leaf, valid marker)` → marker does not rescue; newest fallback repairs both.
6. Dangling, non-USER, and non-root markers → newest fallback and repair.
7. Empty tree + non-null marker → clear both; empty tree + both null → no write.
8. Legacy flat node with `message.parent_message_id is None` remains valid after `_chain_legacy_flat_roots` changes its native parent.
9. Non-empty prompt → `has_user_work=True`; image-only empty-text prompt → empty draft, `has_user_work=False`, no attachment restaging.

```python
def test_valid_before_first_restores_empty_path_and_durable_text() -> None:
    root = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="current durable prompt",
        persisted_message_id="root-user",
        parent_message_id=None,
    )
    reply = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="reply",
        persisted_message_id="reply",
        parent_message_id="root-user",
    )
    store = ConsoleChatStore()
    session = store.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id="conversation",
        all_nodes=[root, reply],
        active_leaf_persisted_id=None,
        active_leaf_before_persisted_id="root-user",
    )
    assert store.active_path_message_ids(session.id) == []
    assert store.active_leaf(session.id) is None
    assert store.session_draft(session.id) == "current durable prompt"
    assert session.has_user_work is True
```

For the legacy authority case, build `[U1, A1, U2]` with all persisted parents null and mark U2. Assert U2's native parent becomes non-null, its `parent_message_id` remains null, and the marker still restores an empty path with draft `"u2"`.

- [ ] **Step 2: Run store tests to verify RED.**

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_console_chat_store_before_first.py -q
```

Expected: FAIL because restore accepts only the scalar active leaf.

- [ ] **Step 3: Extend restore signatures and implement the tri-state resolver.**

Add `active_leaf_before_persisted_id: str | None = None` to `restore_persisted_session` and `_ingest_full_tree`. After node registration and legacy repair:

```python
nodes = self._nodes_by_session.get(session_id, {})
leaf_native = None
repair_cursor = False
restore_before_native = None

if active_leaf_persisted_id is not None:
    leaf_native = persisted_to_native.get(active_leaf_persisted_id)
    if leaf_native is None:
        leaf_native = self._most_recent_leaf_native(session_id)
        repair_cursor = True
    elif active_leaf_before_persisted_id is not None:
        repair_cursor = True
elif active_leaf_before_persisted_id is not None:
    candidate = persisted_to_native.get(active_leaf_before_persisted_id)
    node = nodes.get(candidate) if candidate is not None else None
    if (
        node is not None
        and node.role is ConsoleMessageRole.USER
        and node.parent_message_id is None
    ):
        restore_before_native = candidate
    else:
        leaf_native = self._most_recent_leaf_native(session_id)
        repair_cursor = True
else:
    leaf_native = self._most_recent_leaf_native(session_id)
    repair_cursor = leaf_native is not None

if restore_before_native is not None:
    self._active_leaf_by_session[session_id] = None
    self._recompute_active_path(session_id)
    self.set_session_draft(session_id, nodes[restore_before_native].content)
else:
    self._active_leaf_by_session[session_id] = leaf_native
    self._recompute_active_path(session_id)
    if repair_cursor:
        self._persist_active_leaf(session_id, leaf_native)
```

The active-leaf branch is authoritative even when dangling. `_persist_active_leaf(..., None)` on an invalid empty-tree cursor intentionally clears both columns. Keep context-summary resolution after the path decision.

- [ ] **Step 4: Run Step 2 and verify GREEN.**

- [ ] **Step 5: Write a failing production hydration test.**

```python
@pytest.mark.asyncio
async def test_production_hydration_restores_before_first_cursor(tmp_path) -> None:
    app = _fixture_app(tmp_path)
    assert app.chachanotes_db.set_conversation_active_cursor(
        CONVERSATION_ID,
        active_leaf_message_id=None,
        before_message_id="m1",
    ) is True
    store = app.console_runtime.ensure_chat_store()
    session = await hydrate_console_session(
        app=app,
        store=store,
        conversation_id=CONVERSATION_ID,
        tree=FIXTURE_TREE,
        settings=default_console_session_settings(app.app_config),
    )
    assert store.active_path_message_ids(session.id) == []
    assert store.session_draft(session.id) == "first user message"
    assert session.has_user_work is True
```

- [ ] **Step 6: Run the named hydration test and verify RED.**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_conversation_hydration.py::test_production_hydration_restores_before_first_cursor -q
```

- [ ] **Step 7: Thread the pair through hydration with scalar fallback for old doubles.**

```python
cursor_reader = getattr(db, "get_conversation_active_cursor", None)
if callable(cursor_reader):
    active_leaf_id, active_leaf_before_id = cursor_reader(target)
else:
    active_leaf_id = getattr(
        db, "get_conversation_active_leaf", lambda _target: None
    )(target)
    active_leaf_before_id = None

session = store.restore_persisted_session(
    title=title,
    workspace_id=workspace_id,
    persisted_conversation_id=target,
    all_nodes=all_nodes,
    active_leaf_persisted_id=active_leaf_id,
    active_leaf_before_persisted_id=active_leaf_before_id,
    settings=settings,
    runtime_backend=runtime_backend,
    assistant_kind=assistant_kind,
    assistant_id=assistant_id,
    assistant_authority_id=assistant_authority_id,
    persona_memory_mode=persona_memory_mode,
    character_id=character_id,
    character_name=character_name,
    activate=False,
)
```

- [ ] **Step 8: Run hydration and store regressions; verify GREEN.**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_chat_store_before_first.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_resume_active_path.py -q
```

- [ ] **Step 9: Commit the resume/hydration unit.**

```bash
git add tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_conversation_hydration.py \
  Tests/Chat/test_console_chat_store_before_first.py \
  Tests/Chat/test_console_conversation_hydration.py
git commit -m "feat(console): restore durable before-first position"
```

### Task 4: Wire the first-prompt UI and clear stale markers during acceptance

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:13509-13602`
- Modify: `Tests/UI/test_console_rewind_restore.py:70-125`
- Modify: `tldw_chatbook/Chat/console_dispatch_repository.py:207-224`
- Modify: `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`

- [ ] **Step 1: Write failing screen routing and warning tests.**

Keep the existing mid-path test. Strengthen the first-prompt test with a spy around `set_active_path_before`, then add the false-result behavior:

```python
@pytest.mark.asyncio
async def test_first_prompt_warns_if_restart_cursor_is_unsaved(
    monkeypatch,
) -> None:
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        spy_insert = MagicMock(return_value=True)
        console._insert_prompt_text_into_composer = spy_insert
        original = store.set_active_path_before

        def apply_but_report_unsaved(
            session_id: str, message_id: str
        ) -> bool:
            assert original(session_id, message_id) is True
            return False

        monkeypatch.setattr(
            store, "set_active_path_before", apply_but_report_unsaved
        )
        notices: list[tuple[str, str]] = []
        app.notify = lambda text, **kwargs: notices.append(
            (str(text), kwargs.get("severity", ""))
        )
        await console._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind="restore",
                message_id=ids["u1"].id,
                prompt_text="U1",
            ),
        )
        await pilot.pause()

    assert store.active_path_message_ids(session.id) == []
    spy_insert.assert_called_once_with("U1", replace=True)
    assert (
        "Rewound for this session, but the restart position could not be saved.",
        "warning",
    ) in notices
```

- [ ] **Step 2: Run the screen tests to verify RED.**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_console_rewind_restore.py -q
```

Expected: FAIL because index zero still calls `set_active_leaf(..., None)` and never warns.

- [ ] **Step 3: Route index zero through the dedicated operation.**

```python
if index == 0:
    restart_cursor_saved = store.set_active_path_before(
        session_id, choice.message_id
    )
    if not restart_cursor_saved:
        self.app_instance.notify(
            "Rewound for this session, but the restart position "
            "could not be saved.",
            severity="warning",
        )
else:
    store.set_active_leaf(session_id, path[index - 1])
```

Always continue to fetch the full message text, replace the composer, focus it, and sync the UI. A false result is a warning, not a rollback or early return.

- [ ] **Step 4: Run Step 2 and verify GREEN.**

- [ ] **Step 5: Write the failing direct-acceptance atomic-clear test.**

```python
def test_acceptance_replaces_before_first_marker_atomically(tmp_path: Path) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "cursor.sqlite")
    assert db.set_conversation_active_cursor(
        conversation_id,
        active_leaf_message_id=None,
        before_message_id="old-root",
    ) is True
    inserted = _insert(
        db,
        ConsoleDispatchRepository(db),
        _acceptance(conversation_id),
    )
    assert db.get_conversation_active_cursor(conversation_id) == (
        inserted.assistant_message_id,
        None,
    )
```

- [ ] **Step 6: Run the named checkpoint test to verify RED.**

```bash
../../.venv/bin/python -m pytest \
  Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py::test_acceptance_replaces_before_first_marker_atomically -q
```

Expected: FAIL with the companion marker still populated.

- [ ] **Step 7: Clear the companion in the acceptance transaction.**

Change the existing statement only:

```sql
UPDATE conversations
   SET active_leaf_message_id = ?,
       active_leaf_before_message_id = NULL
 WHERE id = ? AND deleted = 0
```

Keep the existing `rowcount == 1` integrity check. Do not rely on a later best-effort store write.

- [ ] **Step 8: Run UI and checkpoint suites to verify GREEN.**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_rewind_restore.py \
  Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py -q
```

- [ ] **Step 9: Commit the UI/acceptance unit.**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_rewind_restore.py \
  tldw_chatbook/Chat/console_dispatch_repository.py \
  Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py
git commit -m "fix(console): preserve rewind cursor through acceptance"
```

### Task 5: Prove restart, resend, branch recovery, and legacy non-deletion

**Files:**
- Modify: `Tests/UI/test_console_resume_active_path.py:310-345,420-520`
- Modify: `Tests/integration/test_console_rewind_e2e.py:65-125,147-350`

- [ ] **Step 1: Update real-DB hydration helpers to read the cursor pair.**

In both helpers:

```python
active_leaf_id, before_message_id = db.get_conversation_active_cursor(
    conversation_id
)
session = store.restore_persisted_session(
    title="Saved conversation",
    workspace_id=None,
    persisted_conversation_id=conversation_id,
    all_nodes=all_nodes,
    active_leaf_persisted_id=active_leaf_id,
    active_leaf_before_persisted_id=before_message_id,
)
```

- [ ] **Step 2: Correct the existing rewind integration fixture's Library-policy hydration.**

After the first send promotes the temporary session, hydrate its durable authority before the second send:

```python
result1 = await controller.submit_draft("U1")
assert result1.accepted is True
await store.hydrate_session_library_policy(session.id)
result2 = await controller.submit_draft("U2")
```

This is test-fixture plumbing only. Do not modify product policy gates.

- [ ] **Step 3: Add the failing canonical restart lifecycle test.**

```python
@pytest.mark.asyncio
async def test_before_first_survives_restart_then_resend_clears_marker() -> None:
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        store, controller, session, _gateway = _new_controller(db, ["A1"])
        assert (await controller.submit_draft("U1")).accepted is True
        await store.hydrate_session_library_policy(session.id)
        original = store.messages_for_session(session.id)
        root = original[0]
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None

        assert store.set_active_path_before(session.id, root.id) is True
        assert db.get_conversation_active_cursor(conversation_id) == (
            None,
            root.persisted_message_id,
        )

        resumed, resumed_session = _resume_into_fresh_store(db, conversation_id)
        assert resumed.active_path_message_ids(resumed_session.id) == []
        assert resumed.session_draft(resumed_session.id) == "U1"
        resumed.set_session_draft(resumed_session.id, "U1 edited")
        await resumed.hydrate_session_library_policy(resumed_session.id)

        resumed_controller = ConsoleChatController(
            store=resumed,
            provider_gateway=_SequencedCapturingGateway(["A1 edited"]),
        )
        assert (await resumed_controller.submit_draft("U1 edited")).accepted is True
        active_leaf, before = db.get_conversation_active_cursor(conversation_id)
        assert active_leaf is not None
        assert before is None

        restarted, restarted_session = _resume_into_fresh_store(
            db, conversation_id
        )
        assert [
            message.content
            for message in restarted.messages_for_session(restarted_session.id)
        ] == ["U1 edited", "A1 edited"]
        active_root = restarted.messages_for_session(restarted_session.id)[0]
        roots, _index, count = restarted.siblings_at(active_root.id)
        assert count == 2
        old_root = next(root for root in roots if root.content == "U1")
        restarted.set_active_leaf(
            restarted_session.id,
            restarted._leaf_under(old_root.id),
        )
        assert [
            message.content
            for message in restarted.messages_for_session(restarted_session.id)
        ] == ["U1", "A1"]
    finally:
        db.close_connection()
```

Also add the session-only draft case: after first restart, edit or clear the hydrated draft, drop the store without sending, reopen, and assert the durable prompt text appears again.

- [ ] **Step 4: Add real-DB invalid and legacy repair tests.**

In `Tests/UI/test_console_resume_active_path.py` add explicit tests for:

- valid leaf + marker: leaf wins and DB becomes `(leaf, None)`;
- dangling leaf + valid marker: newest fallback wins and marker clears;
- invalid marker + empty tree: DB becomes `(None, None)`;
- legacy flat conversation rewound before U1, followed by one new persisted root and restart: every original row plus the new row still exists; assert durable row IDs/count, not a sibling shape;
- marker target content changed before hydration: restored draft uses the loaded row's current content, proving the ID is not a snapshot.

- [ ] **Step 5: Run the new integration and real-DB verification.**

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_console_resume_active_path.py \
  Tests/integration/test_console_rewind_e2e.py -q
```

Expected: PASS because Tasks 1–4 already implement the unit-level behavior. If a new end-to-end test fails, trace the real integration gap and fix only that omission; do not force an artificial RED state or broaden scope.

- [ ] **Step 6: Re-run Step 5 and verify GREEN.**

- [ ] **Step 7: Commit the lifecycle regression unit.**

```bash
git add Tests/UI/test_console_resume_active_path.py \
  Tests/integration/test_console_rewind_e2e.py
git commit -m "test(console): cover before-first restart lifecycle"
```

### Task 6: Focused verification, review, and Backlog completion

**Files:**
- Modify: `backlog/tasks/task-574 - Console-rewind-restore-to-before-first-message-not-restart-durable.md`
- Review: every file in the responsibility map

- [ ] **Step 1: Run the complete focused regression set.**

```bash
../../.venv/bin/python -m pytest \
  Tests/DB/test_chachanotes_v54_before_first_cursor.py \
  Tests/DB/test_chachanotes_active_leaf_migration.py \
  Tests/DB/test_chachanotes_v53_safe_capture_trim.py \
  Tests/DB/test_chachanotes_console_thinking_migration.py \
  Tests/Chat/test_console_chat_store_before_first.py \
  Tests/Chat/test_console_chat_store_tree.py \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/UI/test_console_rewind_restore.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py \
  Tests/integration/test_console_rewind_e2e.py -q
```

Expected: PASS. Do not run the full repository suite unless the user explicitly opts in, per `AGENTS.md`.

- [ ] **Step 2: Run derived-artifact and whitespace checks.**

```bash
PYTHON=../../.venv/bin/python ./scripts/preflight.sh
git diff --check
```

Expected: every preflight check passes and `git diff --check` emits no output.

- [ ] **Step 3: Perform a code-grounded self-review.**

Inspect `git diff origin/dev...HEAD` and verify:

- cursor columns are always written atomically;
- scalar getter/setter signatures remain compatible;
- no Sync trigger/payload, export/import, fork, or trajectory format includes the marker;
- valid active leaf wins over a contradictory marker;
- dangling active leaf does not fall through to a valid marker;
- pre-repair `message.parent_message_id`, not repaired native parent, decides root status;
- persisted failure returns false after the in-memory path is cleared;
- UI warning does not return before composer refill;
- acceptance SQL clears the marker in the same transaction;
- logs contain IDs/status only, never prompt content.

- [ ] **Step 4: Update TASK-574 acceptance criteria and implementation notes.**

Check all five acceptance criteria only after focused evidence is green. Add concise notes naming the migration/API, store/hydration behavior, UI warning, atomic acceptance clear, legacy limitation, focused command, and ADR-098. Add a lesson only if implementation produces a genuine reusable incident.

```bash
backlog task edit 574 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
  -s Done \
  --notes "Added the v54 local explicit-before-first cursor, atomic cursor API, store/hydration resume semantics, honest UI warning, and acceptance-time marker clear. Covered canonical restart/resend, legacy non-deletion, invalid repair, temporary sessions, and attachment-only text behavior with focused tests. ADR: backlog/decisions/098-console-active-path-before-first-cursor.md."
backlog task 574 --plain
```

- [ ] **Step 5: Repeat Steps 1–2 after documentation edits.**

Expected: PASS and clean whitespace.

- [ ] **Step 6: Commit completion metadata.**

```bash
git add 'backlog/tasks/task-574 - Console-rewind-restore-to-before-first-message-not-restart-durable.md'
git commit -m "docs(task-574): record implementation evidence"
```

- [ ] **Step 7: Apply verification-before-completion.**

Use `@superpowers:verification-before-completion`, confirm `git status --short` is empty, and confirm `origin/dev` remains an ancestor of `HEAD`. If `dev` advanced during implementation, rebase before final verification and rerun the focused evidence.
