# Roleplay P2g-2 — Console world-book inspector + attach/detach — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a "World Books" block to the Console run inspector showing the conversation's attached world books, with Attach/Detach actions — mirroring the merged dict inspector (P1g).

**Architecture:** A `summarize_active_world_books` resolver; `world_book_rows`/`world_book_actions` on `ConsoleInspectorState` rendered as a trailing block; a cached scope-guarded off-thread refresh; and two `console-io` workers driving `WorldBookPicker` + `associate/disassociate_world_book_with_conversation`.

**Tech Stack:** Python 3.11+, Textual, SQLite (`CharactersRAGDB`), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-21-roleplay-p2g-2-console-worldbook-inspector-design.md`.

## Global Constraints

- **No schema migration** (v22). **No send-path change** (P2g-1 already applies attached books on native send). Legacy `chat_events.py` gate-fix + dead-code deletion is P2g-3 — not this cycle.
- The summary shows **ALL attached books** (enabled + disabled, marked) via `get_world_books_for_conversation(conv_id, enabled_only=False)` — NOT the send-path `_collect_active_world_books` (enabled-only).
- The world-book block is a **trailing custom block** (like the dict block), threaded through **all three** `ConsoleRunInspector` spots: `compose()`, `_rendered_row_entries()`, `_structural_key()`. **No `_ROW_ID_BY_LABEL`/`_ROW_GROUPS` change.**
- Row/action projection + the action-list method read **ONLY the cache + the conversation id — never the DB**. The DB read happens only in the scope-guarded off-thread `refresh_active_world_books_summary()`.
- Conversation-only on native (`char_data=None`). Reuse `WorldBookPicker` (P2f) for both Attach and Detach. Leave the dict inspector, its refresh, and its actions untouched.
- Action workers: off-thread, every await individually guarded (uncaught worker exception kills the app under `run_worker(exit_on_error=True)`), `finally`-reset the dialog flag.
- **Subagents:** first verify `git rev-parse --show-toplevel` == the worktree + branch `claude/roleplay-p2g-2-inspector` (Agent-tool subagents can start in the MAIN checkout). Run tests **foreground, scoped** — NO background jobs, NO broad sweeps. Stage ONLY the task's files; never `git add -A`; never stage `.superpowers/`.
- **Test env** (venv in MAIN checkout; run from the worktree root):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: `summarize_active_world_books` resolver

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_info_resolver.py`
- Test: `Tests/Character_Chat/test_summarize_active_world_books.py` (new)

**Interfaces:**
- Produces: `summarize_active_world_books(db, conversation_id: str | None, char_data: dict | None) -> dict` → `{"world_books": [{"name": str, "enabled": bool, "entry_count": int}], "source": "local"}`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Character_Chat/test_summarize_active_world_books.py`:
```python
import pytest

from tldw_chatbook.Character_Chat.world_info_resolver import (
    summarize_active_world_books,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "summarize_wb.db", "test-client")
    yield db
    db.close_connection()


def test_lists_attached_books_with_counts(wb_db):
    wb_db.add_conversation({"id": "c1", "title": "C"})
    wb = WorldBookManager(wb_db)
    b1 = wb.create_world_book("Alpha")
    wb.create_world_book_entry(b1, keys=["a"], content="x")
    wb.create_world_book_entry(b1, keys=["b"], content="y")
    wb.associate_world_book_with_conversation("c1", b1)
    out = summarize_active_world_books(wb_db, "c1", None)
    assert out["world_books"] == [{"name": "Alpha", "enabled": True, "entry_count": 2}]


def test_includes_disabled_attached_book(wb_db):
    wb_db.add_conversation({"id": "c2", "title": "C"})
    wb = WorldBookManager(wb_db)
    b = wb.create_world_book("Off", enabled=False)
    wb.create_world_book_entry(b, keys=["k"], content="x")
    wb.associate_world_book_with_conversation("c2", b)
    out = summarize_active_world_books(wb_db, "c2", None)
    assert out["world_books"] == [{"name": "Off", "enabled": False, "entry_count": 1}]


def test_empty_when_none_attached(wb_db):
    wb_db.add_conversation({"id": "c3", "title": "C"})
    assert summarize_active_world_books(wb_db, "c3", None) == {"world_books": [], "source": "local"}


def test_no_conversation_returns_empty(wb_db):
    assert summarize_active_world_books(wb_db, None, None) == {"world_books": [], "source": "local"}


def test_db_error_returns_empty():
    assert summarize_active_world_books(object(), "cX", None) == {"world_books": [], "source": "local"}
```
(If `get_conversation_by_id` doesn't exist, drop the helper's guard — just call `add_conversation` once per test as the other tests do.)

- [ ] **Step 2: Run to verify they fail**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_summarize_active_world_books.py -q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — `ImportError: cannot import name 'summarize_active_world_books'`.

- [ ] **Step 3: Add the function**

In `world_info_resolver.py`, add after `apply_world_info_to_message` (and add `summarize_active_world_books` to `__all__`):
```python
def summarize_active_world_books(
    db: Any,
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """World-book "what's in play" summary for the Console inspector (never raises).

    Shows ALL attached books (enabled + disabled) — the *attachment picture* —
    via ``get_world_books_for_conversation(enabled_only=False)``, NOT the
    send-path ``_collect_active_world_books`` (which is enabled-only). ``char_data``
    is accepted for signature parity but P2g-2 is conversation-only (native
    ``char_data=None``).

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: Unused on native Console (accepted for parity).

    Returns:
        ``{"world_books": [{"name": str, "enabled": bool, "entry_count": int}],
        "source": "local"}``. ``{"world_books": [], "source": "local"}`` on no
        conversation / no books / any error.
    """
    if not conversation_id or db is None:
        return {"world_books": [], "source": "local"}
    try:
        from .world_book_manager import WorldBookManager

        books = WorldBookManager(db).get_world_books_for_conversation(
            str(conversation_id), enabled_only=False
        )
    except Exception:
        logger.opt(exception=True).debug(
            "world-info: could not summarize conversation world books"
        )
        return {"world_books": [], "source": "local"}
    world_books = []
    for book in books:
        if not isinstance(book, dict):
            continue
        entries = book.get("entries")
        world_books.append(
            {
                "name": str(book.get("name") or "Unnamed"),
                "enabled": bool(book.get("enabled", True)),
                "entry_count": len(entries) if isinstance(entries, list) else 0,
            }
        )
    return {"world_books": world_books, "source": "local"}
```

- [ ] **Step 4: Run to verify they pass**

Run the Step-2 command. Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Character_Chat/world_info_resolver.py Tests/Character_Chat/test_summarize_active_world_books.py
git commit -m "feat(lore): summarize_active_world_books for the Console inspector"
```

---

### Task 2: Inspector state field + trailing render block

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (add two fields to `ConsoleInspectorState`)
- Modify: `tldw_chatbook/Widgets/Console/console_run_inspector.py` (render block in 3 spots)
- Test: `Tests/UI/test_console_run_inspector_worldbooks.py` (new)

**Interfaces:**
- Consumes: `ConsoleDisplayRow(label, value)`, `ConsoleInspectorAction(widget_id, label, enabled, disabled_reason)`.
- Produces: `ConsoleInspectorState.world_book_rows` / `.world_book_actions`; a "World Books" block with row ids `console-inspector-worldbooks-row-{index}` and heading id `console-inspector-worldbooks-heading`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_console_run_inspector_worldbooks.py`. Read `Tests/UI/` for how `ConsoleRunInspector` is mounted with a `ConsoleInspectorState` in existing inspector tests (search for `ConsoleRunInspector(` / `dictionary_rows` in Tests/) and mirror. The test builds a `ConsoleInspectorState` with `world_book_rows=(ConsoleDisplayRow("Alpha", "2 entries"),)` and `world_book_actions=(ConsoleInspectorAction("console-inspector-worldbooks-attach","Attach world book…",enabled=True),)`, mounts the inspector, and asserts: the "World Books" heading Static exists; a `#console-inspector-worldbooks-row-0` Static with text `"Alpha: 2 entries"`; the attach button `#console-inspector-worldbooks-attach` exists; and with empty tuples the block is absent. Also assert the dict block still renders when `dictionary_rows` is set (no regression).

- [ ] **Step 2: Run to verify it fails**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_console_run_inspector_worldbooks.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL (no world_book_rows field / no rendered block).

- [ ] **Step 3: Add the state fields**

In `console_display_state.py`, in `ConsoleInspectorState`, after `dictionary_rows`/`dictionary_actions`:
```python
    world_book_rows: tuple[ConsoleDisplayRow, ...] = ()
    world_book_actions: tuple[ConsoleInspectorAction, ...] = ()
```

- [ ] **Step 4: Render the block (all three spots) in `console_run_inspector.py`**

(a) In `compose()`, immediately after the `dict_rows`/`dict_actions` block, add:
```python
        world_book_rows = getattr(self.state, "world_book_rows", ())
        world_book_actions = getattr(self.state, "world_book_actions", ())
        if world_book_rows or world_book_actions:
            yield Static(
                "World Books",
                id="console-inspector-worldbooks-heading",
                classes="console-inspector-group-heading destination-section",
            )
            for index, row in enumerate(world_book_rows):
                yield Static(
                    row.text,
                    id=f"console-inspector-worldbooks-row-{index}",
                    classes=f"console-inspector-row console-inspector-row-{row.status}",
                    markup=False,
                )
            for action in world_book_actions:
                yield from self._compose_action(action)
```
(b) In `_rendered_row_entries()`, after the `dictionary_rows` loop:
```python
        for index, row in enumerate(getattr(state, "world_book_rows", ()) or ()):
            entries.append(
                (f"console-inspector-worldbooks-row-{index}", row.text, row.status)
            )
```
(c) In `_structural_key()`, add a third tuple to the returned tuple (after the `dictionary_actions` one):
```python
            tuple(
                _action_key(action)
                for action in getattr(state, "world_book_actions", ()) or ()
            ),
```

- [ ] **Step 5: Run to verify + no regression**

Run the Step-2 command (PASS). Then run the existing inspector tests to confirm no regression:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/ -k "run_inspector" -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

- [ ] **Step 6: Commit**
```bash
git add tldw_chatbook/Chat/console_display_state.py tldw_chatbook/Widgets/Console/console_run_inspector.py Tests/UI/test_console_run_inspector_worldbooks.py
git commit -m "feat(console): World Books inspector block (trailing render, 3 spots)"
```

---

### Task 3: Screen read wiring (cache, scope-guarded refresh, row projection)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_worldbook_inspector.py` (new)

**Interfaces:**
- Consumes: `summarize_active_world_books` (Task 1); `world_book_rows` (Task 2).
- Produces: `_active_world_books_summary` cache; `refresh_active_world_books_summary()`; `_refresh_active_world_books_summary_if_scope_changed()`; `_active_console_world_book_scope_ids()`; `_console_world_book_inspector_rows()`; `world_book_rows=…` wired into `_build_console_inspector_state`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_console_worldbook_inspector.py`. Read the existing dict-inspector screen test (search Tests/ for `_console_dictionary_inspector_rows` / `refresh_active_dictionaries_summary` / `_active_dictionaries_summary`) and mirror its harness. The test: builds the ChatScreen with a real `CharactersRAGDB`; pins a native session to a conversation with two attached world books (`WorldBookManager.associate_world_book_with_conversation`); calls `await screen.refresh_active_world_books_summary()`; asserts `screen._console_world_book_inspector_rows()` yields rows for both books (`.text` contains each name); and `_build_console_inspector_state()` carries them in `world_book_rows`. Also: with no conversation → a single "No active chat" row; conversation but no books → "No world books in play".

- [ ] **Step 2: Run to verify it fails**

Same command shape as Task 2 Step 2 for the new file. Expected: FAIL (methods missing).

- [ ] **Step 3: Add the cache + scope-ids + refresh + scope-guard**

In `chat_screen.py.__init__`, next to the dict cache init (`self._active_dictionaries_summary` / `self._last_console_dictionary_scope_ids`):
```python
        self._active_world_books_summary: dict | None = None
        # Sentinel distinct from any real `(conversation_id,)` tuple so the
        # first scope check always refreshes.
        self._last_console_world_book_scope_ids: tuple | None = None
```
Add these methods near the dict ones (`_active_console_dictionary_scope_ids` / `refresh_active_dictionaries_summary` / `_refresh_active_dictionaries_summary_if_scope_changed`):
```python
    def _active_console_world_book_scope_ids(self) -> tuple[str | None]:
        """(conversation_id,) for the active native Console world-book scope.

        Conversation-only: native Console has no character (see
        `_active_console_dictionary_scope_ids`).
        """
        return (self._current_console_rail_conversation_id(),)

    async def refresh_active_world_books_summary(self) -> None:
        """The ONLY place that performs the DB-backed world-book summarize.

        `_build_console_inspector_state` reads only the cache set here. The
        sync summarize is marshalled onto a worker thread via `asyncio.to_thread`
        so it never blocks the UI loop.
        """
        conversation_id = self._current_console_rail_conversation_id()
        db = getattr(self.app_instance, "chachanotes_db", None)
        if not conversation_id or db is None:
            self._active_world_books_summary = {"world_books": []}
            self._sync_console_control_bar()
            return
        try:
            from ...Character_Chat.world_info_resolver import (
                summarize_active_world_books,
            )

            summary = await asyncio.to_thread(
                summarize_active_world_books, db, conversation_id, None
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Could not summarize active world books for the Console inspector."
            )
            summary = {"world_books": []}
        self._active_world_books_summary = (
            summary if isinstance(summary, dict) else {"world_books": []}
        )
        self._sync_console_control_bar()

    async def _refresh_active_world_books_summary_if_scope_changed(self) -> None:
        """Recompute the world-book summary only when the scope changed."""
        scope_ids = self._active_console_world_book_scope_ids()
        if scope_ids == self._last_console_world_book_scope_ids:
            return
        self._last_console_world_book_scope_ids = scope_ids
        await self.refresh_active_world_books_summary()

    def _console_world_book_inspector_rows(self) -> tuple[ConsoleDisplayRow, ...]:
        """Project the cached world-book summary into inspector rows (no DB)."""
        conversation_id = self._current_console_rail_conversation_id()
        if conversation_id is None:
            return (ConsoleDisplayRow("No active chat", ""),)
        summary = self._active_world_books_summary or {}
        books = summary.get("world_books") or []
        if not books:
            return (ConsoleDisplayRow("No world books in play", ""),)
        rows = []
        for entry in books:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "Unnamed")
            count = entry.get("entry_count")
            value = f"{count} entries" if isinstance(count, int) else "0 entries"
            if not entry.get("enabled", True):
                value += " (disabled)"
            rows.append(ConsoleDisplayRow(name, value))
        return tuple(rows)
```
(Confirm `_sync_console_control_bar` is the method the dict refresh calls at its end; match whatever the dict refresh does.)

- [ ] **Step 4: Wire the rows into `_build_console_inspector_state` + the scope-guard into `_sync_native_console_chat_ui`**

In `_build_console_inspector_state()`, in the `replace(inspector_state, dictionary_rows=…, dictionary_actions=…)` call, add:
```python
            world_book_rows=self._console_world_book_inspector_rows(),
```
In `_sync_native_console_chat_ui()`, immediately after `await self._refresh_active_dictionaries_summary_if_scope_changed()`, add:
```python
            await self._refresh_active_world_books_summary_if_scope_changed()
```

- [ ] **Step 5: Run to verify + no regression**

Run the new test (PASS), then the dict-inspector screen test(s) to confirm no regression (find them via the search in Step 1).

- [ ] **Step 6: Commit**
```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_worldbook_inspector.py
git commit -m "feat(console): populate World Books inspector rows (cached, scope-guarded)"
```

---

### Task 4: Attach/Detach actions + workers

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_worldbook_inspector.py` (extend Task 3's file)

**Interfaces:**
- Consumes: `world_book_actions` (Task 2); `refresh_active_world_books_summary` (Task 3); `WorldBookPicker` (P2f); `WorldBookManager.get_world_books_for_conversation`/`associate_world_book_with_conversation`/`disassociate_world_book_from_conversation`/`list_world_books`.
- Produces: `_console_world_book_inspector_actions()`; `_console_worldbook_dialog_active`; two `Button.Pressed` handlers; `_console_worldbook_attach_worker`/`_console_worldbook_detach_worker`; `world_book_actions=…` wired into `_build_console_inspector_state`.

- [ ] **Step 1: Write the failing tests**

Extend `Tests/UI/test_console_worldbook_inspector.py` (mirror the dict Console attach/detach test — search Tests/ for `_console_dictionary_attach_worker` / `console-inspector-dictionaries-attach`): with a native session pinned to a conversation and a standalone world book seeded, monkeypatch `app_instance.push_screen_wait` to return the book id for a `WorldBookPicker`, click/post the attach action → `WorldBookManager(db).get_world_books_for_conversation(conv_id)` shows it + the inspector rows update; then detach → removed. Also assert `_console_world_book_inspector_actions()` gates Attach on a conversation id and Detach on ≥1 attached book.

- [ ] **Step 2: Run to verify it fails** (methods/handlers missing).

- [ ] **Step 3: Add the actions projection + dialog flag + handlers + workers**

Add the dialog-flag init in `__init__` (next to `_console_dictionary_dialog_active`):
```python
        self._console_worldbook_dialog_active = False
```
Add the actions method (near `_console_dictionary_inspector_actions`):
```python
    def _console_world_book_inspector_actions(
        self,
    ) -> tuple[ConsoleInspectorAction, ...]:
        """Attach/Detach actions for the Console world-book block (cache + conv id only)."""
        conversation_id = self._current_console_rail_conversation_id()
        summary = self._active_world_books_summary or {}
        has_attached = bool(summary.get("world_books"))
        return (
            ConsoleInspectorAction(
                "console-inspector-worldbooks-attach",
                "Attach world book…",
                enabled=bool(conversation_id),
                disabled_reason="Start or load a conversation first",
            ),
            ConsoleInspectorAction(
                "console-inspector-worldbooks-detach",
                "Detach world book…",
                enabled=has_attached,
            ),
        )
```
Wire it into `_build_console_inspector_state`'s `replace(...)` alongside `world_book_rows`:
```python
            world_book_actions=self._console_world_book_inspector_actions(),
```
Add the two `Button.Pressed` handlers (next to the dict inspector ones at `on_console_inspector_dictionaries_attach`/`_detach`):
```python
    @on(Button.Pressed, "#console-inspector-worldbooks-attach")
    def on_console_inspector_worldbooks_attach(self, event: Button.Pressed) -> None:
        event.stop()
        if self._console_worldbook_dialog_active:
            return
        self._console_worldbook_dialog_active = True
        self.run_worker(self._console_worldbook_attach_worker(), group="console-io")

    @on(Button.Pressed, "#console-inspector-worldbooks-detach")
    def on_console_inspector_worldbooks_detach(self, event: Button.Pressed) -> None:
        event.stop()
        if self._console_worldbook_dialog_active:
            return
        self._console_worldbook_dialog_active = True
        self.run_worker(self._console_worldbook_detach_worker(), group="console-io")
```
Add the workers (mirror `_console_dictionary_attach_worker`/`_detach_worker`; import `WorldBookPicker` + `WorldBookManager` lazily inside the workers):
```python
    async def _console_worldbook_attach_worker(self) -> None:
        try:
            conversation_id = self._current_console_rail_conversation_id()
            if not conversation_id:
                self.app_instance.notify("Start or load a conversation first.", severity="warning")
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                return
            from ...Character_Chat.world_book_manager import WorldBookManager
            from ...Widgets.Persona_Widgets.world_book_picker import WorldBookPicker

            def _attachable() -> list[dict]:
                mgr = WorldBookManager(db)
                attached_ids = {b.get("id") for b in mgr.get_world_books_for_conversation(str(conversation_id), enabled_only=False)}
                return [
                    {"world_book_id": int(b.get("id")), "name": str(b.get("name"))}
                    for b in (mgr.list_world_books(include_disabled=False) or [])
                    if b.get("id") not in attached_ids
                ]
            try:
                rows = await asyncio.to_thread(_attachable)
            except Exception:
                logger.opt(exception=True).warning("Could not load world books for the Console attach picker.")
                return
            if not rows:
                self.app_instance.notify("No more world books to attach.", severity="information")
                return
            try:
                picked = await self.app_instance.push_screen_wait(WorldBookPicker(rows))
            except Exception:
                logger.opt(exception=True).warning("Could not show the Console world-book picker.")
                return
            if not picked:
                return
            try:
                await asyncio.to_thread(
                    WorldBookManager(db).associate_world_book_with_conversation,
                    str(conversation_id), int(picked),
                )
            except Exception as exc:
                logger.opt(exception=True).warning("Could not attach the world book.")
                self.app_instance.notify(f"Attach failed: {exc}", severity="error")
                return
            await self.refresh_active_world_books_summary()
        finally:
            self._console_worldbook_dialog_active = False

    async def _console_worldbook_detach_worker(self) -> None:
        try:
            conversation_id = self._current_console_rail_conversation_id()
            if not conversation_id:
                self.app_instance.notify("Start or load a conversation first.", severity="warning")
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                return
            from ...Character_Chat.world_book_manager import WorldBookManager
            from ...Widgets.Persona_Widgets.world_book_picker import WorldBookPicker

            def _attached() -> list[dict]:
                mgr = WorldBookManager(db)
                return [
                    {"world_book_id": int(b.get("id")), "name": str(b.get("name"))}
                    for b in mgr.get_world_books_for_conversation(str(conversation_id), enabled_only=False)
                ]
            try:
                rows = await asyncio.to_thread(_attached)
            except Exception:
                logger.opt(exception=True).warning("Could not load world books for the Console detach picker.")
                return
            if not rows:
                self.app_instance.notify("No world books attached to this conversation.", severity="information")
                return
            try:
                picked = await self.app_instance.push_screen_wait(
                    WorldBookPicker(rows, title="Detach world book", confirm_label="Detach")
                )
            except Exception:
                logger.opt(exception=True).warning("Could not show the Console world-book picker.")
                return
            if not picked:
                return
            try:
                await asyncio.to_thread(
                    WorldBookManager(db).disassociate_world_book_from_conversation,
                    str(conversation_id), int(picked),
                )
            except Exception as exc:
                logger.opt(exception=True).warning("Could not detach the world book.")
                self.app_instance.notify(f"Detach failed: {exc}", severity="error")
                return
            await self.refresh_active_world_books_summary()
        finally:
            self._console_worldbook_dialog_active = False
```
Confirm `Button` and `@on` are imported in `chat_screen.py` (the dict handlers use them). `disassociate_world_book_from_conversation(conversation_id, world_book_id)` — verify arg order against `world_book_manager.py` (P2e: `(conversation_id, world_book_id)`).

- [ ] **Step 4: Run to verify + full P2g-2 gate + app import**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_summarize_active_world_books.py Tests/UI/test_console_run_inspector_worldbooks.py \
Tests/UI/test_console_worldbook_inspector.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('APP IMPORT OK')"
```

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_worldbook_inspector.py
git commit -m "feat(console): World Books inspector Attach/Detach actions + workers"
```

---

## Notes for reviewers

- **No migration; no send-path change.** Legacy `chat_events.py` gate-fix + dead-code deletion is P2g-3.
- The summary uses `enabled_only=False` (attachment picture, shows disabled marked) — NOT the send-path `_collect_active_world_books`.
- The world-book block is a **trailing custom block** threaded through `compose`/`_rendered_row_entries`/`_structural_key` — no `_ROW_ID_BY_LABEL` change; the dict block is untouched.
- Projection/action methods read ONLY the cache + conversation id; the DB read is only in the scope-guarded off-thread refresh.
- Action workers guard every await + `finally`-reset `_console_worldbook_dialog_active`; conversation-only; `WorldBookPicker` reused for Attach and Detach.
