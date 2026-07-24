# Roleplay P3d-1 — Reactive Console Character Expression Avatar — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Console Character-rail avatar (P3c, #782) swap among idle/thinking/speaking/error images as the character generates and streams a reply, using per-state images stored locally on the character.

**Architecture:** A new `character_expression_images` BLOB table (migration v22→v23) holds non-idle images; idle reuses `character_cards.image`. A pure DB-free function derives the current state from the store's in-memory message statuses. The P3c avatar scope-guard widens from `(character_id,)` to `(character_id, state)`, so the *already-wired* `_sync_native_console_chat_ui` refresh (0.2s poll + transition syncs) drives the swap — no new hook. All P3c render/refresh machinery is reused; a per-state decode cache makes revisiting a state instant. Authoring adds three upload slots to the character editor.

**Tech Stack:** Python ≥3.11, Textual, SQLite (CharactersRAGDB, threading.local connections), PIL/rich_pixels/textual_image, loguru.

## Global Constraints

- **MIGRATION v22→v23**: idempotent (`CREATE TABLE IF NOT EXISTS`); **pre-flight re-verify `_CURRENT_SCHEMA_VERSION` is still 22** at plan+merge time (concurrent sessions may take v23). Only DB change; `character_cards` untouched (idle stays `character_cards.image`). New tables live ONLY in the migration, never the frozen base CREATE.
- Resolve active character ONLY off the live `ConsoleChatSession`; resolve expression state ONLY off in-memory store status — **NEVER a DB read on the derivation path** (it runs on the 0.2s tick).
- REUSE the single `self._console_image_cache` + `resolve_default_mode` + `fit_image_cell_size` + the P3c render/refresh methods. Do NOT create a second render cache. Decode stays OFF-THREAD (`asyncio.to_thread`).
- **NEVER raise** into `_sync_native_console_chat_ui`; preserve EVERY P3c invariant (config-off early-return + `_last_console_avatar_scope=None` reset, post-await re-check — now on `(character_id, state)`, cache-as-SPEC-not-widget, scope-guard = no rebuild/decode except on a real state change).
- **DO NOT touch** `_active_console_dictionary_scope_ids` (P3c pin — feeds the "what's in play" summaries; not inert).
- Authoring slots apply the `_character_editor_generation` render-token discipline PER slot; expression saves write straight to the table, **independent of the card's optimistic-lock version**.
- Config-gated: `[chat.images].show_character_avatar` (section) + `[chat.images].react_character_expressions` (swapping), BOTH default **True**. `react` off ⇒ derivation pins state to `idle` (P3c avatar still renders); `show_character_avatar` off ⇒ section cleared entirely.
- Characters-only (personas have no image). State-id validation lives in the Python seam, not a DB CHECK (keeps the table state-agnostic for future custom states).
- **CONCURRENT-SESSION HAZARD**: `chat_screen.py` / `personas_screen.py` / `ChaChaNotes_DB.py` are heavily edited by other sessions — keep P3d localized; expect a rebase before merge.
- Implementers stage ONLY their task's files (never `git add -A`, never `.superpowers/`). NO background/broad test sweeps; NEVER broad-pkill pytest — scope to the worktree.
- `Tests/UI/pytest.ini` sets `asyncio_mode=auto`: keep async tests in `Tests/UI/` OR add explicit `@pytest.mark.asyncio`; do NOT mix `Tests/UI` with another dir in one pytest invocation (rootdir shift drops auto-mode).
- **Test env prefix** (all pytest + import runs): `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ... -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`
- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign`. Subagents PREPEND `cd <worktree> && ` to EVERY Bash call (cwd doesn't persist; main checkout is on another branch).
- **DB tests use a file-backed `CharactersRAGDB(tmp_path / "...db", "test-client")`, NEVER `:memory:`** — the avatar fetch runs off-thread and `:memory:` is connection-private (a worker thread would see an empty DB).

---

## File Structure

- `tldw_chatbook/DB/ChaChaNotes_DB.py` — **modify**: new `character_expression_images` table via a v22→v23 migration + 4 CRUD methods. (Task 1)
- `tldw_chatbook/Chat/console_expression_state.py` — **create**: pure `resolve_console_expression_state` + the state constants. (Task 2)
- `tldw_chatbook/Chat/console_image_view.py` — **modify**: add `resolve_react_character_expressions`. (Task 2)
- `tldw_chatbook/UI/Screens/chat_screen.py` — **modify**: widen the P3c avatar scope to `(character_id, state)`, add the per-state decode cache + table-fetch-with-fallback. (Task 3)
- `tldw_chatbook/UI/Screens/personas_screen.py` (+ its CSS) — **modify**: 3 expression upload slots in the character editor. (Task 4)
- `tldw_chatbook/config.py` — **modify**: document `react_character_expressions` in the `[chat.images]` block. (Task 5)

---

## Task 1: `character_expression_images` table + migration + CRUD seam

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Test: `Tests/Character_Chat/test_character_expression_images.py` (create)

**Interfaces:**
- Produces (methods on `CharactersRAGDB`):
  - `set_character_expression_image(character_id: int, state_id: str, image: bytes, mime: str | None = None) -> None`
  - `get_character_expression_image(character_id: int, state_id: str) -> bytes | None`
  - `list_character_expression_states(character_id: int) -> list[str]`
  - `delete_character_expression_image(character_id: int, state_id: str) -> None`
  - Module constant `_EXPRESSION_IMAGE_STATE_IDS = frozenset({"thinking", "speaking", "error"})` (validation set).

- [ ] **Step 1: PRE-FLIGHT — verify the schema version is still 22**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && grep -n "_CURRENT_SCHEMA_VERSION = " tldw_chatbook/DB/ChaChaNotes_DB.py
```
Expected: `_CURRENT_SCHEMA_VERSION = 22`. **If it is NOT 22** (another session took v23), STOP and report BLOCKED — the migration target must become v23→v24 and the plan needs re-basing.

- [ ] **Step 2: Write the failing test**

Create `Tests/Character_Chat/test_character_expression_images.py`:
```python
import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    # File-backed (NOT :memory:) — CharactersRAGDB uses threading.local connections
    # and the production avatar fetch runs off-thread; :memory: is connection-private.
    return CharactersRAGDB(tmp_path / "expr.db", "test-client")


def _make_character(db) -> int:
    return db.add_character_card({"name": "Ada"})


def test_schema_is_v23(db):
    assert db._get_db_version(db.get_connection()) == 23


def test_set_then_get_round_trips(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "speaking", b"PNGBYTES", "image/png")
    assert db.get_character_expression_image(cid, "speaking") == b"PNGBYTES"


def test_get_missing_returns_none(db):
    cid = _make_character(db)
    assert db.get_character_expression_image(cid, "thinking") is None


def test_set_is_upsert_on_character_and_state(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "speaking", b"one")
    db.set_character_expression_image(cid, "speaking", b"two")
    assert db.get_character_expression_image(cid, "speaking") == b"two"
    assert db.list_character_expression_states(cid) == ["speaking"]  # not duplicated


def test_list_states_returns_only_active(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "thinking", b"t")
    db.set_character_expression_image(cid, "error", b"e")
    assert sorted(db.list_character_expression_states(cid)) == ["error", "thinking"]


def test_delete_is_soft_and_get_returns_none(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "speaking", b"s")
    db.delete_character_expression_image(cid, "speaking")
    assert db.get_character_expression_image(cid, "speaking") is None
    assert db.list_character_expression_states(cid) == []


def test_set_after_delete_reactivates(db):
    cid = _make_character(db)
    db.set_character_expression_image(cid, "speaking", b"s1")
    db.delete_character_expression_image(cid, "speaking")
    db.set_character_expression_image(cid, "speaking", b"s2")
    assert db.get_character_expression_image(cid, "speaking") == b"s2"


def test_invalid_state_id_rejected(db):
    cid = _make_character(db)
    with pytest.raises(ValueError):
        db.set_character_expression_image(cid, "idle", b"x")   # idle is never stored
    with pytest.raises(ValueError):
        db.set_character_expression_image(cid, "bogus", b"x")
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_character_expression_images.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `test_schema_is_v23` fails (version is 22) and `AttributeError: 'CharactersRAGDB' object has no attribute 'set_character_expression_image'`.

- [ ] **Step 4: Add the migration SQL constant**

In `ChaChaNotes_DB.py`, next to the other `_MIGRATE_V*_SQL` constants (e.g. just after `_MIGRATE_V21_TO_V22_SQL`), add:
```python
    # Keep this runner SQL aligned with
    # tldw_chatbook/DB/migrations/chachanotes_v22_to_v23_character_expression_images.sql.
    _MIGRATE_V22_TO_V23_SQL = """
CREATE TABLE IF NOT EXISTS character_expression_images(
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  character_id  INTEGER NOT NULL REFERENCES character_cards(id) ON DELETE CASCADE ON UPDATE CASCADE,
  state_id      TEXT    NOT NULL,
  image         BLOB    NOT NULL,
  mime          TEXT,
  created_at    TEXT    NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW')),
  updated_at    TEXT    NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW')),
  deleted       INTEGER NOT NULL DEFAULT 0,
  UNIQUE(character_id, state_id)
);
CREATE INDEX IF NOT EXISTS idx_char_expr_images_char ON character_expression_images(character_id);
UPDATE db_schema_version
   SET version = 23
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 22;
"""
```

- [ ] **Step 5: Add the migration method**

Add just after `_migrate_from_v21_to_v22` (mirror its shape; no `PRAGMA table_info` column-guard is needed because `CREATE TABLE IF NOT EXISTS` is already idempotent):
```python
    def _migrate_from_v22_to_v23(self, conn: sqlite3.Connection):
        """Migrate schema V22→V23: add the local ``character_expression_images``
        BLOB table (per-state reaction avatars; idle reuses character_cards.image)."""
        logger.info(f"Migrating schema from V22 to V23 for '{self._SCHEMA_NAME}' in DB: {self.db_path_str}...")
        try:
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

- [ ] **Step 6: Register the step + bump the version**

In the `migration_steps` dict (the ladder), add the new entry after `21: self._migrate_from_v21_to_v22,`:
```python
                    22: self._migrate_from_v22_to_v23,
```
Change the version constant:
```python
    _CURRENT_SCHEMA_VERSION = 23  # Adds character_expression_images (P3d reaction avatars).
```

- [ ] **Step 7: Add the CRUD seam**

Add near the other character-card methods (e.g. after `get_character_card_by_id`). `_EXPRESSION_IMAGE_STATE_IDS` goes at module scope near the top of the class file:
```python
_EXPRESSION_IMAGE_STATE_IDS = frozenset({"thinking", "speaking", "error"})
```
```python
    def set_character_expression_image(
        self, character_id: int, state_id: str, image: bytes, mime: str | None = None
    ) -> None:
        """Upsert a per-state expression image for a character (state-agnostic
        store; ``idle`` is never stored here -- it reuses character_cards.image)."""
        if state_id not in _EXPRESSION_IMAGE_STATE_IDS:
            raise ValueError(f"Unknown expression state_id: {state_id!r}")
        if not isinstance(image, (bytes, bytearray)) or not image:
            raise ValueError("Expression image must be non-empty bytes.")
        query = """
            INSERT INTO character_expression_images(character_id, state_id, image, mime, deleted, updated_at)
            VALUES (?, ?, ?, ?, 0, STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW'))
            ON CONFLICT(character_id, state_id) DO UPDATE SET
                image = excluded.image,
                mime = excluded.mime,
                deleted = 0,
                updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW')
        """
        with self.transaction() as cur:
            cur.execute(query, (character_id, state_id, bytes(image), mime))

    def get_character_expression_image(self, character_id: int, state_id: str) -> bytes | None:
        """Return the active image bytes for (character, state), or None."""
        cur = self.get_connection().execute(
            "SELECT image FROM character_expression_images "
            "WHERE character_id = ? AND state_id = ? AND deleted = 0",
            (character_id, state_id),
        )
        row = cur.fetchone()
        return bytes(row[0]) if row is not None and row[0] is not None else None

    def list_character_expression_states(self, character_id: int) -> list[str]:
        """Return the state_ids with an active image for a character."""
        cur = self.get_connection().execute(
            "SELECT state_id FROM character_expression_images "
            "WHERE character_id = ? AND deleted = 0 ORDER BY state_id",
            (character_id,),
        )
        return [row[0] for row in cur.fetchall()]

    def delete_character_expression_image(self, character_id: int, state_id: str) -> None:
        """Soft-delete the (character, state) expression image."""
        with self.transaction() as cur:
            cur.execute(
                "UPDATE character_expression_images SET deleted = 1, "
                "updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW') "
                "WHERE character_id = ? AND state_id = ?",
                (character_id, state_id),
            )
```
NOTE: match the codebase's actual transaction/cursor idiom — verify whether the class exposes `self.transaction()` (context manager) or a different helper by reading how `add_character_card` / `update_character_card` execute writes, and mirror that exact idiom for the two write methods above. Reads use `self.get_connection().execute(...)` as shown.

- [ ] **Step 8: Run the tests to verify they pass**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_character_expression_images.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (8 tests). Also confirm the app imports:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app"
```

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py Tests/Character_Chat/test_character_expression_images.py
git commit -m "feat(db): P3d-1 Task 1 — character_expression_images table (v22->v23) + CRUD seam"
```

---

## Task 2: `resolve_console_expression_state` + `resolve_react_character_expressions`

**Files:**
- Create: `tldw_chatbook/Chat/console_expression_state.py`
- Modify: `tldw_chatbook/Chat/console_image_view.py`
- Test: `Tests/Chat/test_console_expression_state.py` (create)

**Interfaces:**
- Consumes: the store's `messages_for_session(session_id) -> list[ConsoleChatMessage]` (in-memory) and `ConsoleMessageRole.ASSISTANT` / `ConsoleMessageStatus` from `tldw_chatbook.Chat.console_chat_models`. Status values: `Literal["complete","pending","streaming","stopped","failed"]`.
- Produces:
  - `EXPRESSION_STATES: tuple = ("idle", "thinking", "speaking", "error")`
  - `EXPRESSION_IMAGE_STATES: tuple = ("thinking", "speaking", "error")`
  - `resolve_console_expression_state(store, active_session_id: str | None, *, react_enabled: bool) -> str`
  - `resolve_react_character_expressions(app_config: Mapping[str, Any]) -> bool` (in `console_image_view.py`)

- [ ] **Step 1: Write the failing test**

Create `Tests/Chat/test_console_expression_state.py`:
```python
import pytest
from tldw_chatbook.Chat.console_expression_state import (
    EXPRESSION_STATES,
    EXPRESSION_IMAGE_STATES,
    resolve_console_expression_state,
)
from tldw_chatbook.Chat.console_image_view import resolve_react_character_expressions


class _Msg:
    def __init__(self, role, status):
        self.role = role
        self.status = status


class _FakeRole:
    ASSISTANT = object()
    USER = object()


class _FakeStore:
    """Minimal stand-in exposing messages_for_session, matching the real signature."""
    def __init__(self, messages_by_session):
        self._m = messages_by_session

    def messages_for_session(self, session_id):
        if session_id not in self._m:
            raise KeyError(session_id)
        return list(self._m[session_id])


@pytest.fixture(autouse=True)
def _patch_role(monkeypatch):
    # Point the resolver at the fake role sentinel so _Msg.role comparisons match.
    import tldw_chatbook.Chat.console_expression_state as mod
    monkeypatch.setattr(mod, "ConsoleMessageRole", _FakeRole)


def _state(messages, *, react=True, sid="s1"):
    store = _FakeStore({sid: messages})
    return resolve_console_expression_state(store, sid, react_enabled=react)


def test_no_session_is_idle():
    store = _FakeStore({})
    assert resolve_console_expression_state(store, None, react_enabled=True) == "idle"


def test_missing_session_is_idle():
    store = _FakeStore({})
    assert resolve_console_expression_state(store, "nope", react_enabled=True) == "idle"


def test_no_assistant_message_is_idle():
    assert _state([_Msg(_FakeRole.USER, "complete")]) == "idle"


def test_pending_assistant_is_thinking():
    assert _state([_Msg(_FakeRole.USER, "complete"), _Msg(_FakeRole.ASSISTANT, "pending")]) == "thinking"


def test_streaming_assistant_is_speaking():
    assert _state([_Msg(_FakeRole.ASSISTANT, "streaming")]) == "speaking"


def test_complete_assistant_is_idle():
    assert _state([_Msg(_FakeRole.ASSISTANT, "complete")]) == "idle"


def test_stopped_assistant_is_idle():
    assert _state([_Msg(_FakeRole.ASSISTANT, "stopped")]) == "idle"


def test_failed_assistant_is_error():
    assert _state([_Msg(_FakeRole.ASSISTANT, "failed")]) == "error"


def test_last_assistant_wins():
    # A completed turn followed by a new pending turn -> thinking.
    msgs = [_Msg(_FakeRole.ASSISTANT, "complete"), _Msg(_FakeRole.ASSISTANT, "pending")]
    assert _state(msgs) == "thinking"


def test_react_disabled_pins_idle():
    assert _state([_Msg(_FakeRole.ASSISTANT, "streaming")], react=False) == "idle"


def test_constants():
    assert EXPRESSION_STATES == ("idle", "thinking", "speaking", "error")
    assert EXPRESSION_IMAGE_STATES == ("thinking", "speaking", "error")


def test_react_config_helper_defaults_true():
    assert resolve_react_character_expressions({}) is True
    cfg = {"chat": {"images": {"react_character_expressions": False}}}
    assert resolve_react_character_expressions(cfg) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_expression_state.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Chat.console_expression_state` / `resolve_react_character_expressions` not found.

- [ ] **Step 3: Create the derivation module**

Create `tldw_chatbook/Chat/console_expression_state.py`:
```python
"""Pure, DB-free derivation of the Console 'expression state' for the reactive
character avatar (P3d). Reads only the store's in-memory message statuses -- safe
to call on the 0.2s transcript poll tick.

State machine (from the active session's last assistant message status):
  pending   -> "thinking"   (created, awaiting first token)
  streaming -> "speaking"   (tokens flowing)
  complete  -> "idle"
  stopped   -> "idle"       (user stop is not an error)
  failed    -> "error"
  (no assistant message / no session / react disabled) -> "idle"
"""
from __future__ import annotations

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

EXPRESSION_STATES = ("idle", "thinking", "speaking", "error")
EXPRESSION_IMAGE_STATES = ("thinking", "speaking", "error")

_STATUS_TO_STATE = {
    "pending": "thinking",
    "streaming": "speaking",
    "complete": "idle",
    "stopped": "idle",
    "failed": "error",
}


def resolve_console_expression_state(store, active_session_id, *, react_enabled: bool) -> str:
    """Return the current expression state for the active Console session.

    Args:
        store: the ConsoleChatStore (exposes ``messages_for_session``).
        active_session_id: the live session id, or None.
        react_enabled: whether reactive swapping is enabled; when False, always idle.

    Returns:
        One of EXPRESSION_STATES. Never raises (any lookup failure -> "idle").
    """
    if not react_enabled or active_session_id is None or store is None:
        return "idle"
    try:
        messages = store.messages_for_session(active_session_id)
    except Exception:
        return "idle"
    last_assistant = None
    for message in messages:  # transcript order; keep the last assistant turn
        if getattr(message, "role", None) is ConsoleMessageRole.ASSISTANT:
            last_assistant = message
    if last_assistant is None:
        return "idle"
    return _STATUS_TO_STATE.get(getattr(last_assistant, "status", "complete"), "idle")
```

- [ ] **Step 4: Add the config helper**

In `tldw_chatbook/Chat/console_image_view.py`, immediately after `resolve_show_character_avatar`, add (mirroring it exactly):
```python
def resolve_react_character_expressions(app_config: Mapping[str, Any]) -> bool:
    """Whether the Console avatar reacts (swaps images) as the character
    thinks/speaks (default True). Reads ``[chat.images].react_character_expressions``
    via the same both-shapes accessor as ``resolve_show_character_avatar``.

    Returns:
        True unless explicitly disabled via ``react_character_expressions = false``.
    """
    value = _chat_images_config(app_config).get("react_character_expressions", True)
    return bool(value)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_expression_state.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (12 tests).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_expression_state.py tldw_chatbook/Chat/console_image_view.py Tests/Chat/test_console_expression_state.py
git commit -m "feat(console): P3d-1 Task 2 — expression-state derivation + react config helper"
```

---

## Task 3: Widen the P3c avatar scope to `(character_id, state)` + per-state cache + table fetch

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (the P3c avatar refresh; `_refresh_active_character_avatar_if_scope_changed` ~:3935 — READ FRESH, lines drift)
- Test: `Tests/UI/test_console_character_avatar.py` (extend)

**Interfaces:**
- Consumes: `resolve_console_expression_state` + `EXPRESSION_IMAGE_STATES` (Task 2); `resolve_react_character_expressions` (Task 2); `db.get_character_expression_image` (Task 1). The P3c methods `_current_console_rail_character_id/_name`, `_active_native_console_session`, `_ensure_console_image_view`, `_fetch_character_card_for_avatar`, `_build_character_avatar_widget`, `_render_character_avatar_into_section`, and the fields `_active_character_avatar`, `_active_character_avatar_name`, `_last_console_avatar_scope`, `self._console_image_cache`, `self._console_image_default_mode`, `CHARACTER_AVATAR_COLS`/`CHARACTER_AVATAR_LINES`.
- Produces: a `(character_id, state)`-keyed decode cache; the refresh now swaps on state change.

**READ FIRST at plan time:** the current body of `_refresh_active_character_avatar_if_scope_changed`, `_render_character_avatar_into_section`, `_build_character_avatar_widget`, `_fetch_character_card_for_avatar`, and the P3c cache-field inits. The steps below describe the *edits* to make; preserve every existing P3c guard.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_character_avatar.py` (reuse the existing `console_screen_with_db` fixture + `_set_active_console_character` helper). This drives the state via a fake store the screen already exposes, OR by setting message statuses on the real controller store. Use the store the screen's controller exposes:
```python
@pytest.mark.asyncio
async def test_avatar_swaps_across_expression_states(console_screen_with_db, monkeypatch):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    def _png(color):
        buf = BytesIO(); PILImage.new("RGB", (32, 32), color).save(buf, format="PNG"); return buf.getvalue()
    char_id = db.add_character_card({"name": "Ada", "image": _png((10, 10, 10))})
    db.set_character_expression_image(char_id, "speaking", _png((0, 200, 0)))
    _set_active_console_character(screen, char_id, "Ada")

    # Drive the derived state directly (the pure resolver is unit-tested separately);
    # here we assert the refresh reacts to the state it computes.
    import tldw_chatbook.UI.Screens.chat_screen as cs
    state_box = {"v": "idle"}
    monkeypatch.setattr(cs, "resolve_console_expression_state", lambda *a, **k: state_box["v"])

    state_box["v"] = "idle"
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None
    assert screen._last_console_avatar_scope == (char_id, "idle")

    state_box["v"] = "speaking"
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope == (char_id, "speaking")

    # Revisiting a state is served from the per-state cache (no re-decode).
    assert (char_id, "speaking") in screen._console_expression_spec_cache


@pytest.mark.asyncio
async def test_expression_state_falls_back_to_idle_image(console_screen_with_db, monkeypatch):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32), (5, 5, 5)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})   # idle image only
    _set_active_console_character(screen, char_id, "Ada")
    import tldw_chatbook.UI.Screens.chat_screen as cs
    monkeypatch.setattr(cs, "resolve_console_expression_state", lambda *a, **k: "thinking")

    await screen._refresh_active_character_avatar_if_scope_changed()   # no thinking image -> idle image
    assert screen._active_character_avatar is not None   # rendered the idle fallback, did not crash
    assert screen._last_console_avatar_scope == (char_id, "thinking")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_character_avatar.py::test_avatar_swaps_across_expression_states -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — scope is `(char_id,)` not `(char_id, "idle")`, and `_console_expression_spec_cache` does not exist.

- [ ] **Step 3: Add the imports + the per-state cache field**

At the top of `chat_screen.py` (with the other P3c imports), add:
```python
from tldw_chatbook.Chat.console_expression_state import (
    resolve_console_expression_state,
    EXPRESSION_IMAGE_STATES,
)
from tldw_chatbook.Chat.console_image_view import resolve_react_character_expressions
```
(`resolve_show_character_avatar` / `fit_image_cell_size` are already imported from P3c.) In `__init__`, next to the P3c avatar cache fields, add:
```python
        self._console_expression_spec_cache: dict[tuple[int, str], dict] = {}
```

- [ ] **Step 4: Add a state-aware fetch helper**

Add next to `_fetch_character_card_for_avatar`:
```python
    def _fetch_expression_image_bytes(self, character_id: int, state: str) -> bytes | None:
        """Return the image bytes for (character, state): the expression-table
        image for a non-idle state, else the character's idle avatar. Runs
        off-thread (called via asyncio.to_thread). Never raises -> None on any error."""
        try:
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                return None
            if state in EXPRESSION_IMAGE_STATES:
                img = db.get_character_expression_image(character_id, state)
                if img:
                    return img
            card = self._fetch_character_card_for_avatar(character_id)  # idle fallback
            image = (card or {}).get("image")
            return bytes(image) if isinstance(image, (bytes, bytearray)) and image else None
        except Exception:
            logger.opt(exception=True).debug("avatar: expression fetch failed")
            return None
```

- [ ] **Step 5: Rewrite the refresh to be state-aware**

Edit `_refresh_active_character_avatar_if_scope_changed` so that (preserving every existing P3c guard):
1. The `show_character_avatar` config-off branch is unchanged (early-return, clear `_active_character_avatar`/`_name`, reset `_last_console_avatar_scope = None`).
2. Compute the state and scope:
```python
        character_id = self._current_console_rail_character_id()
        controller = getattr(self, "_console_chat_controller", None)
        store = getattr(controller, "store", None) if controller is not None else None
        active_session_id = getattr(store, "active_session_id", None) if store is not None else None
        react = resolve_react_character_expressions(
            getattr(getattr(self, "app_instance", None), "app_config", {}) or {}
        )
        state = resolve_console_expression_state(store, active_session_id, react_enabled=react)
        scope = (character_id, state)
        if scope == self._last_console_avatar_scope:
            return
        self._last_console_avatar_scope = scope
        name = self._current_console_rail_character_name()
        self._active_character_avatar_name = name
        if character_id is None:
            self._active_character_avatar = None
            await self._render_character_avatar_into_section()
            return
```
3. Serve from the per-state cache when present (no decode):
```python
        cached = self._console_expression_spec_cache.get((character_id, state))
        if cached is not None:
            self._active_character_avatar = cached
            await self._render_character_avatar_into_section()
            return
```
4. Otherwise decode off-thread (mirror the P3c fetch/prepare, but via `_fetch_expression_image_bytes` and a state-scoped cache key `f"character:{character_id}:{state}"`), build the spec, then **post-await re-check the full `(character_id, state)` scope** (recompute state) before storing/rendering — drop a stale render if the scope moved:
```python
        _, cache = self._ensure_console_image_view()
        mode = getattr(self, "_console_image_default_mode", "pixels")
        key = f"character:{character_id}:{state}"
        spec = {"character_id": character_id, "state": state, "name": name,
                "mode": mode, "pil": None, "pixels": None}
        try:
            image = await asyncio.to_thread(self._fetch_expression_image_bytes, character_id, state)
            if image:
                ok = await asyncio.to_thread(cache.prepare, key, image)
                if ok:
                    spec["pil"] = cache.get_pil(key)
                    if mode != "graphics":
                        spec["pixels"] = cache.get_pixels(key)
        except Exception:
            logger.opt(exception=True).debug("avatar: expression decode failed")
        # Post-await staleness re-check on the FULL (character_id, state) scope:
        # the state can flip mid-decode while streaming, so recompute it from
        # the SAME store/session captured above and drop a stale render.
        current_state = resolve_console_expression_state(store, active_session_id, react_enabled=react)
        if (self._current_console_rail_character_id(), current_state) != scope or not self.is_mounted:
            return
        self._console_expression_spec_cache[(character_id, state)] = spec
        self._active_character_avatar = spec
        await self._render_character_avatar_into_section()
```
NOTE: keep the exact `cache.prepare`/`get_pil`/`get_pixels`/`_ensure_console_image_view`/`fit_image_cell_size` calls identical to the current P3c body — only the fetch source (`_fetch_expression_image_bytes`), the cache key (state-scoped), the scope tuple, and the per-state spec cache are new. Do NOT change `_build_character_avatar_widget` (it already reads `spec["pil"]`/`spec["pixels"]`/`spec["mode"]`; the extra `"state"` key is ignored) and do NOT change `_render_character_avatar_into_section`.

- [ ] **Step 6: Run the tests to verify they pass**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_character_avatar.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (all existing P3c tests + the 2 new ones). The pre-existing P3c tests still assert the old `(char_id,)` scope in some places — UPDATE those assertions to `(char_id, "idle")` where they check `_last_console_avatar_scope` (the generic/idle path now yields `(char_id, "idle")`; a None character yields `(None, "idle")`). Do NOT weaken any never-raise / config-off test.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_character_avatar.py
git commit -m "feat(console): P3d-1 Task 3 — reactive avatar scope (character_id,state) + per-state cache + table fetch"
```

---

## Task 4: Expression authoring slots in the character editor

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (+ its `.tcss` if the avatar thumb has dedicated CSS — mirror the avatar-thumb rule for the 3 slots)
- Test: `Tests/UI/test_personas_expression_slots.py` (create)

**Interfaces:**
- Consumes: `db.set_character_expression_image` / `get_character_expression_image` / `delete_character_expression_image` (Task 1); the existing editor avatar machinery: `_character_editor_generation`, `_avatar_upload_dialog_worker` (~:4156), `_fit_avatar_cell_size` (~:4034), `_build_avatar_pixels` (~:4071), and the ConsoleImageRenderCache + `resolve_default_mode` off-thread pattern.
- Produces: three per-state upload/preview/clear slots (thinking/speaking/error) that write directly to the expression table, independent of the card's optimistic-lock version.

**READ FIRST at plan time:** the current avatar-slot compose + `_avatar_upload_dialog_worker` + `_render_character_editor_avatar` (the method around :3960-4032 that reads `editor.current_avatar_bytes()`, captures the generation token, decodes off-thread, and re-checks the token) to mirror the shape for each expression slot.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_personas_expression_slots.py` — mount the personas screen editor for a saved character, assert the 3 slots exist and that uploading writes a row / clearing soft-deletes it. Mirror the existing personas editor test harness (find it with `grep -rln "personas_screen\|PersonasScreen\|character editor" Tests/UI`). Skeleton:
```python
import pytest
# ... import the personas screen + its harness exactly as the existing editor tests do ...

@pytest.mark.asyncio
async def test_expression_slots_present_for_saved_character(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    for state in ("thinking", "speaking", "error"):
        assert screen.query_one(f"#char-expression-slot-{state}") is not None


@pytest.mark.asyncio
async def test_upload_writes_expression_row(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (16, 16)).save(buf, format="PNG")
    await screen._apply_expression_upload(char_id, "speaking", buf.getvalue(), "image/png")
    assert db.get_character_expression_image(char_id, "speaking") is not None


@pytest.mark.asyncio
async def test_clear_soft_deletes_expression_row(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    db.set_character_expression_image(char_id, "error", b"x")
    await screen._clear_expression_slot(char_id, "error")
    assert db.get_character_expression_image(char_id, "error") is None
```
(Adjust method names to the ones you implement in Step 3; the test asserts the write-through behavior, not internal wiring.)

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_expression_slots.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — the slots and the `_apply_expression_upload` / `_clear_expression_slot` methods do not exist.

- [ ] **Step 3: Add the slots + write-through handlers**

In the character editor compose (next to the avatar thumbnail), add an "Expressions" area containing three slots, each `id="char-expression-slot-{state}"` with an upload button, a thumbnail `Static`, and a clear button. Implement:
- `_apply_expression_upload(character_id: int, state: str, image: bytes, mime: str | None) -> None`: `db.set_character_expression_image(...)`, then bump `self._character_editor_generation` and re-render THAT slot's thumbnail (mirror `_render_character_editor_avatar`'s token-capture + off-thread decode + post-await token re-check, per slot).
- `_clear_expression_slot(character_id: int, state: str) -> None`: `db.delete_character_expression_image(...)`, bump the generation, clear that slot's thumbnail to the empty hint.
- An upload worker per slot mirroring `_avatar_upload_dialog_worker` (reuse the same dialog; on accept, call `_apply_expression_upload`).
- On editor open for a saved character, load each slot's existing image via `db.get_character_expression_image` and render its thumbnail.
- **Slots are enabled only for a SAVED character (has an id).** For a brand-new unsaved character, show the slots disabled with a hint "Save the character to add expressions." (A new character has no id to attach rows to.)
- Personas mode shows NO expression slots (characters-only).

Each slot's decode uses the SAME `_character_editor_generation` token discipline as the avatar: capture the token before the off-thread decode, and after it re-check `token == self._character_editor_generation and self.is_mounted` before mounting — a slot replaced/cleared mid-decode must drop its stale render.

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_expression_slots.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS. Also run the existing personas editor test module (find it via grep) to confirm no regression, and `python -c "import tldw_chatbook.app"`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_expression_slots.py
# include the .tcss file only if you added an expression-slot CSS rule
git commit -m "feat(personas): P3d-1 Task 4 — character expression authoring slots (thinking/speaking/error)"
```

---

## Task 5: Config documentation + end-to-end integration & fail-soft

**Files:**
- Modify: `tldw_chatbook/config.py` (document `react_character_expressions` in the `[chat.images]` block ~:2740)
- Test: `Tests/UI/test_console_character_avatar.py` (extend with the full-arc + gate + fail-soft integration tests)

**Interfaces:**
- Consumes: everything from Tasks 1-4.

- [ ] **Step 1: Write the failing integration tests**

Append to `Tests/UI/test_console_character_avatar.py`:
```python
@pytest.mark.asyncio
async def test_react_off_pins_idle_even_when_streaming(console_screen_with_db, monkeypatch):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    def _png(c):
        b = BytesIO(); PILImage.new("RGB", (16, 16), c).save(b, format="PNG"); return b.getvalue()
    char_id = db.add_character_card({"name": "Ada", "image": _png((1, 1, 1))})
    db.set_character_expression_image(char_id, "speaking", _png((0, 255, 0)))
    _set_active_console_character(screen, char_id, "Ada")
    app.app_config["chat"] = {"images": {"react_character_expressions": False}}
    import tldw_chatbook.UI.Screens.chat_screen as cs
    # Even if the raw status would say "streaming", react-off must pin idle.
    # (resolve_console_expression_state honors react_enabled=False internally.)
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope == (char_id, "idle")


@pytest.mark.asyncio
async def test_reactive_avatar_never_raises_on_corrupt_expression(console_screen_with_db, monkeypatch):
    app, screen, db = console_screen_with_db
    char_id = db.add_character_card({"name": "Bad"})
    db.set_character_expression_image(char_id, "speaking", b"not-an-image")
    _set_active_console_character(screen, char_id, "Ada")
    import tldw_chatbook.UI.Screens.chat_screen as cs
    monkeypatch.setattr(cs, "resolve_console_expression_state", lambda *a, **k: "speaking")
    # Must not raise into the sync tick even though the image is corrupt.
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope == (char_id, "speaking")
```

- [ ] **Step 2: Run to verify it fails (or passes trivially), then implement the doc**

Run the two tests. `test_reactive_avatar_never_raises_on_corrupt_expression` should already pass (never-raise inherited); `test_react_off_pins_idle_even_when_streaming` should pass once Task 3 is in. If both pass, they are regression guards — proceed. Then document the config key.

- [ ] **Step 3: Document `react_character_expressions`**

In `tldw_chatbook/config.py`, in the `[chat.images]` block (near `show_character_avatar` ~:2740), add a documented default line mirroring the existing style, e.g.:
```python
# react_character_expressions = true   # Swap the Console character avatar among
#   idle/thinking/speaking/error as it generates a reply (requires per-state images
#   on the character; default true). Set false to keep a static avatar.
```
(Match the exact comment/entry style used for `show_character_avatar` in that block — read it and mirror.)

- [ ] **Step 4: Run the focused suite + app import**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_character_avatar.py Tests/UI/test_console_persistent_rails.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Then Chat + DB suites (separate invocation — different rootdir):
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_expression_state.py Tests/Character_Chat/test_character_expression_images.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Then `python -c "import tldw_chatbook.app"`. Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py Tests/UI/test_console_character_avatar.py
git commit -m "feat(console): P3d-1 Task 5 — document react_character_expressions + full-arc & fail-soft tests"
```

---

## Self-Review (author)

- **Spec coverage:** table+migration+CRUD (Task 1) ✓; `resolve_console_expression_state` DB-free + `resolve_react_character_expressions` + react-off-pins-idle (Task 2) ✓; scope `(character_id,state)` + per-state cache + fallback chain state→idle→text + never-raise + post-await re-check on the full tuple, all P3c machinery reused (Task 3) ✓; authoring slots with render-token discipline + independent-of-card-version save (Task 4) ✓; config doc + gate + fail-soft integration (Task 5) ✓. Out-of-scope items (animation/atlas/import/server-MCP/registry/custom states) are in NO task ✓.
- **Type consistency:** `resolve_console_expression_state(store, active_session_id, *, react_enabled)` and `EXPRESSION_IMAGE_STATES`/`EXPRESSION_STATES` are defined in Task 2 and consumed with matching signatures in Task 3. `set/get/list/delete_character_expression_image` signatures match between Task 1 (definition) and Tasks 3/4 (consumption). Scope tuple `(character_id, state)` is consistent across Task 3 and the Task 3/5 test assertions. Cache field `_console_expression_spec_cache: dict[tuple[int,str],dict]` and cache key `f"character:{character_id}:{state}"` are consistent.
- **Placeholder scan:** no TBD/TODO; every code step shows code; the two "READ FIRST at plan time" notes point at named methods to mirror, not vague hand-waving. Task 4's compose is described structurally (slot ids fixed: `char-expression-slot-{state}`) because it mirrors an existing avatar slot the implementer must read — the exact widget tree follows that sibling.
- **Known drift risk:** all `chat_screen.py` / `personas_screen.py` / `ChaChaNotes_DB.py` line numbers are approximate (concurrent edits) — every task says READ FRESH. The migration ladder + `_MIGRATE_V*_SQL` pattern and the `db_schema_version` bump are verified against `_migrate_from_v21_to_v22` / `_MIGRATE_V18_TO_V19_SQL`.
