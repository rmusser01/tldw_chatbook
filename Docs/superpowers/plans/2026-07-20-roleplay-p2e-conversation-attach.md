# Roleplay P2e — Lore conversation-attach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Roleplay-mode Attachments tab so a user can attach/detach a world book to/from a conversation (its entries then participate in that conversation's world-info injection).

**Architecture:** A read-only reverse-query manager method + an I/O-free Attachments tab in the Lore detail widget + a generic conversation picker + screen handlers that call the existing `WorldBookManager` associate/disassociate methods off-thread. Mirrors the merged Dictionaries P1e. The junction and the legacy send-path application already exist.

**Tech Stack:** Python ≥3.11, Textual (Modal/DataTable/TabPane), pytest + pytest-asyncio, SQLite (`CharactersRAGDB` / `WorldBookManager`).

## Global Constraints

- NO migration — the `conversation_world_books` junction already exists.
- NO send-path change — `chat_events.py` already loads attached books into `WorldInfoProcessor`; attaching takes effect immediately on the legacy send.
- `associate_world_book_with_conversation(conversation_id, world_book_id, priority=0)` is an `INSERT OR REPLACE` **upsert** — it never raises `ConflictError`; re-attach is a harmless no-op.
- Conversation ids are **strings** (UUIDs) throughout — the `conversation_id: int` annotation on the manager methods is misleading; always pass `str(...)`.
- The Lore detail widget stays **I/O-free**: `load_attachments(rows)` takes rows; no DB in the widget. All screen DB reads/writes run off-thread via `asyncio.to_thread`, wrapped → `self._notify` on failure, re-entrancy-guarded with `self._io_dialog_active` + `group="personas-io"`.
- **Reuse** the existing generic `_list_attachable_conversations` (personas_screen); do NOT duplicate it. **Leave the merged `DictionaryAttachPicker` untouched.**
- OUT of scope (→ P2g): Console "what's in play" world-book block, native-Console send application, retiring the legacy Chat-sidebar attach UI.
- Implementers stage ONLY their task's files (`git add <paths>`; never `git add -A`; never `.superpowers/`).
- **Test environment (from the worktree root; UI tests are slow ~60s — allow generous timeouts):**
  ```bash
  HOME=/private/tmp/tldw-chatbook-test-home \
  XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```
  The venv is in the MAIN checkout; `import tldw_chatbook` resolves to the worktree source (cwd on `sys.path`).

## File Structure

- **Modify** `tldw_chatbook/Character_Chat/world_book_manager.py` — add `get_conversations_for_world_book`.
- **Create** `tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py` — generic `ConversationAttachPicker`.
- **Modify** `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` — Attachments tab + messages + `load_attachments`.
- **Modify** `tldw_chatbook/UI/Screens/personas_screen.py` — refresh + attach + detach handlers.
- Tests: `Tests/Character_Chat/test_world_book_manager.py`, `Tests/UI/test_personas_lore.py`.

---

### Task 1: Reverse query `get_conversations_for_world_book`

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_book_manager.py` (add a method after `get_world_books_for_conversation`)
- Test: `Tests/Character_Chat/test_world_book_manager.py`

**Interfaces:**
- Consumes: existing `associate_world_book_with_conversation(conversation_id, world_book_id, priority=0)` (upsert).
- Produces: `get_conversations_for_world_book(world_book_id: int) -> List[Dict[str, Any]]` returning `[{"conversation_id": str, "title": str}]`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Character_Chat/test_world_book_manager.py`:
```python
def test_get_conversations_for_world_book_round_trips(wb_manager):
    db = wb_manager.db
    db.add_conversation({"id": "conv-1", "title": "First case"})
    db.add_conversation({"id": "conv-2", "title": None})  # NULL title
    book_id = wb_manager.create_world_book("B")
    wb_manager.associate_world_book_with_conversation("conv-1", book_id)
    wb_manager.associate_world_book_with_conversation("conv-2", book_id)
    rows = wb_manager.get_conversations_for_world_book(book_id)
    by_id = {r["conversation_id"]: r["title"] for r in rows}
    assert by_id == {"conv-1": "First case", "conv-2": "(untitled)"}
    assert all(isinstance(r["conversation_id"], str) for r in rows)


def test_get_conversations_for_world_book_empty_when_unattached(wb_manager):
    book_id = wb_manager.create_world_book("B")
    assert wb_manager.get_conversations_for_world_book(book_id) == []
```

- [ ] **Step 2: Run to verify they fail**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_manager.py -k "get_conversations_for_world_book" \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `AttributeError: 'WorldBookManager' object has no attribute 'get_conversations_for_world_book'`.

- [ ] **Step 3: Add the method**

In `world_book_manager.py`, add right after `get_world_books_for_conversation`:
```python
    def get_conversations_for_world_book(
        self, world_book_id: int
    ) -> List[Dict[str, Any]]:
        """Conversations this world book is attached to (reverse of
        get_world_books_for_conversation).

        Args:
            world_book_id: The world book to find attachments for.

        Returns:
            ``[{"conversation_id": str, "title": str}]`` (NULL title → "(untitled)").
        """
        query = """
        SELECT cwb.conversation_id, c.title
        FROM conversation_world_books cwb
        JOIN conversations c ON c.id = cwb.conversation_id
        WHERE cwb.world_book_id = ? AND c.deleted = 0
        ORDER BY c.last_modified DESC
        """
        with self.db.transaction() as cursor:
            cursor.execute(query, (world_book_id,))
            return [
                {"conversation_id": str(row[0]), "title": row[1] or "(untitled)"}
                for row in cursor.fetchall()
            ]
```

- [ ] **Step 4: Run to verify they pass**

Run the Step-2 command. Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/world_book_manager.py Tests/Character_Chat/test_world_book_manager.py
git commit -m "feat(lore): WorldBookManager.get_conversations_for_world_book (reverse attach query)"
```

---

### Task 2: Generic `ConversationAttachPicker`

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py`
- Test: `Tests/UI/test_conversation_attach_picker.py`

**Interfaces:**
- Produces: `ConversationAttachPicker(conversations: list[dict[str, Any]])` — a `ModalScreen[str | None]` that dismisses with the picked conversation id (str) or `None`. Ids kept as strings.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_conversation_attach_picker.py`:
```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.conversation_attach_picker import (
    ConversationAttachPicker,
)


class _PickerHost(App):
    def __init__(self, convs):
        super().__init__()
        self._convs = convs
        self.result = "unset"

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.result = await self.push_screen_wait(ConversationAttachPicker(self._convs))


@pytest.mark.asyncio
async def test_picker_select_returns_string_id():
    convs = [{"conversation_id": "c1", "title": "Alpha"},
             {"conversation_id": "c2", "title": "Beta"}]
    app = _PickerHost(convs)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#conversation-attach-list", ListView).index = 1
        await pilot.pause()
        await pilot.click("#conversation-attach-confirm")
        await pilot.pause()
    assert app.result == "c2"


@pytest.mark.asyncio
async def test_picker_filter_narrows_then_selects():
    # After filtering to "beta", index 0 must be Beta (c2) — proving the filter
    # rebuilt the row-id list. If the filter did nothing, index 0 would be c1.
    convs = [{"conversation_id": "c1", "title": "Alpha"},
             {"conversation_id": "c2", "title": "Beta"}]
    app = _PickerHost(convs)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#conversation-attach-search", Input).value = "beta"
        await pilot.pause()
        app.query_one("#conversation-attach-list", ListView).index = 0
        await pilot.pause()
        await pilot.click("#conversation-attach-confirm")
        await pilot.pause()
    assert app.result == "c2"


@pytest.mark.asyncio
async def test_picker_cancel_returns_none():
    convs = [{"conversation_id": "c1", "title": "Alpha"}]
    app = _PickerHost(convs)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#conversation-attach-cancel")
        await pilot.pause()
    assert app.result is None
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_conversation_attach_picker.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError: ...conversation_attach_picker`.

- [ ] **Step 3: Create the picker**

Create `tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py` (generic clone of the merged `DictionaryAttachPicker`, generic name + ids):
```python
"""A small modal for picking a conversation (by string id) to attach the
current entity to. Generic — used by the Roleplay Lore Attachments flow (P2e);
the dictionary flow keeps its own DictionaryAttachPicker.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class ConversationAttachPicker(ModalScreen[str | None]):
    """Pick one conversation (by string id) to attach the current entity to.

    Args:
        conversations: ``{"conversation_id": str, "title": str}`` rows to choose from.
    """

    DEFAULT_CSS = """
    ConversationAttachPicker { align: center middle; }
    ConversationAttachPicker > Vertical {
        width: 60%; max-width: 80; height: auto; max-height: 80%;
        padding: 1 2; border: round $panel;
    }
    ConversationAttachPicker #conversation-attach-list { height: auto; max-height: 16; }
    """

    def __init__(self, conversations: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._conversations = list(conversations)
        self._row_ids: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Attach to conversation", markup=False)
            yield Input(placeholder="Search conversations…", id="conversation-attach-search")
            yield ListView(id="conversation-attach-list")
            with Vertical(id="conversation-attach-actions"):
                yield Button("Attach", id="conversation-attach-confirm", classes="console-action-secondary")
                yield Button("Cancel", id="conversation-attach-cancel", classes="console-action-secondary")

    def on_mount(self) -> None:
        self._populate(self._conversations)

    def _populate(self, rows: list[dict[str, Any]]) -> None:
        listing = self.query_one("#conversation-attach-list", ListView)
        listing.clear()
        self._row_ids = []
        for row in rows:
            item = ListItem(Static(str(row.get("title") or "(untitled)"), markup=False))
            listing.append(item)
            self._row_ids.append(str(row.get("conversation_id")))
        # A filter change must require an explicit re-select.
        listing.index = None

    @on(Input.Changed, "#conversation-attach-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        rows = (
            [c for c in self._conversations if needle in str(c.get("title") or "").lower()]
            if needle
            else self._conversations
        )
        self._populate(rows)

    def _selected_id(self) -> str | None:
        listing = self.query_one("#conversation-attach-list", ListView)
        index = listing.index
        if index is None or not 0 <= index < len(self._row_ids):
            return None
        return self._row_ids[index]

    @on(Button.Pressed, "#conversation-attach-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._selected_id())

    @on(Button.Pressed, "#conversation-attach-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


__all__ = ["ConversationAttachPicker"]
```

- [ ] **Step 4: Run to verify it passes**

Run the Step-2 command. Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py Tests/UI/test_conversation_attach_picker.py
git commit -m "feat(personas): generic ConversationAttachPicker modal"
```

---

### Task 3: Lore Attachments tab (widget)

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` (messages after `:71`; Attachments TabPane after `:210`; `on_mount` `:213-221`; `clear` `:302`; `__all__` `:601`; new `load_attachments` + `_selected_attachment_id` + button handlers)
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: nothing new (`Static`, `DataTable`, `Button`, `TabPane`, `Text`, `Message`, `on` already imported).
- Produces: message classes `LoreAttachRequested` (bare) and `LoreDetachRequested(conversation_id: str)`; `load_attachments(rows: list[dict]) -> None`; ids `#personas-lore-tab-attachments`, `#personas-lore-attachments-empty`, `#personas-lore-attachments-table`, `#personas-lore-attach-add`, `#personas-lore-attach-detach`.

- [ ] **Step 1: Write the failing tests**

The `_DetailHost` harness in `Tests/UI/test_personas_lore.py` only captures `LoreEntryAddRequested` today. Add capture for the new messages: in `_DetailHost.__init__` add `self.attach_posts = []` and `self.detach_posts = []`, and add two handlers to the class:
```python
    def on_lore_attach_requested(self, message) -> None:
        self.attach_posts.append(message)

    def on_lore_detach_requested(self, message) -> None:
        self.detach_posts.append(message.conversation_id)
```
Add the import to the test file's `personas_lore_detail` import block: `LoreAttachRequested, LoreDetachRequested`. Then append these tests:
```python
@pytest.mark.asyncio
async def test_attachments_empty_state_and_render():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        widget.load_attachments([])
        await pilot.pause()
        empty = app.query_one("#personas-lore-attachments-empty", Static)
        table = app.query_one("#personas-lore-attachments-table", DataTable)
        assert empty.display is True and table.row_count == 0
        widget.load_attachments([{"conversation_id": "c1", "title": "Noir case"}])
        await pilot.pause()
        assert empty.display is False and table.row_count == 1


@pytest.mark.asyncio
async def test_attach_button_posts_request():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-tabs", TabbedContent).active = "personas-lore-tab-attachments"
        await pilot.pause()
        await pilot.click("#personas-lore-attach-add")
        await pilot.pause()
        assert len(app.attach_posts) == 1


@pytest.mark.asyncio
async def test_detach_button_posts_selected_conversation():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        widget.load_attachments([{"conversation_id": "c1", "title": "Noir case"}])
        app.query_one("#personas-lore-tabs", TabbedContent).active = "personas-lore-tab-attachments"
        await pilot.pause()
        app.query_one("#personas-lore-attachments-table", DataTable).move_cursor(row=0)
        await pilot.pause()
        await pilot.click("#personas-lore-attach-detach")
        await pilot.pause()
        assert app.detach_posts == ["c1"]
```
Add `TabbedContent` to the test file's `from textual.widgets import ...` line (it is not currently imported there).

- [ ] **Step 2: Run to verify they fail**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "attachments or attach_button or detach_button" \
-q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — `ImportError` (`LoreAttachRequested` not defined) / `#personas-lore-attachments-empty` not found.

- [ ] **Step 3: Add the message classes**

In `personas_lore_detail.py`, after `LoreBookExportRequested` (`:71`):
```python
class LoreAttachRequested(Message):
    """Intent: attach the selected world book to a conversation (opens a picker)."""


class LoreDetachRequested(Message):
    """Intent: detach the selected world book from a conversation."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__()
        self.conversation_id = conversation_id
```

- [ ] **Step 4: Add the Attachments TabPane**

In `compose`, after the Settings TabPane's Export button block (`:210`) and before `yield Static("", id="personas-lore-status", markup=False)` (`:211`), at the TabPane indentation level:
```python
            with TabPane("Attachments", id="personas-lore-tab-attachments"):
                yield Static(
                    "Not attached to any conversation yet.",
                    id="personas-lore-attachments-empty",
                    markup=False,
                )
                yield DataTable(id="personas-lore-attachments-table", cursor_type="row")
                with Horizontal(classes="personas-lore-form-row"):
                    yield Button(
                        "Attach to conversation…",
                        id="personas-lore-attach-add",
                        classes="console-action-secondary",
                    )
                    yield Button(
                        "Detach",
                        id="personas-lore-attach-detach",
                        classes="console-action-secondary",
                    )
```

- [ ] **Step 5: Register the table columns + clear on reset**

In `on_mount` (`:219-220`), after the entries `add_columns`:
```python
        self.query_one("#personas-lore-attachments-table", DataTable).add_columns(
            "conversation", "id"
        )
```
In `clear` (`:302`), add a line resetting attachments (place it with the other clears):
```python
        self.load_attachments([])
```

- [ ] **Step 6: Add `load_attachments` + `_selected_attachment_id` + button handlers**

Add these methods (place `load_attachments` near the other public render methods; the handlers near the other `@on(Button.Pressed, ...)` handlers):
```python
    def load_attachments(self, rows: list[dict]) -> None:
        """Render the conversations this world book is attached to (I/O-free).

        Args:
            rows: ``{"conversation_id": str, "title": str}`` entries.
        """
        self._attachment_rows = list(rows)
        table = self.query_one("#personas-lore-attachments-table", DataTable)
        table.clear()
        for row in self._attachment_rows:
            table.add_row(
                Text(str(row.get("title") or "(untitled)")),
                Text(str(row.get("conversation_id") or "")),
                key=str(row.get("conversation_id")),
            )
        empty = self.query_one("#personas-lore-attachments-empty", Static)
        empty.display = not self._attachment_rows
        table.display = bool(self._attachment_rows)

    def _selected_attachment_id(self) -> str | None:
        table = self.query_one("#personas-lore-attachments-table", DataTable)
        if table.row_count == 0 or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            return str(table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value)
        except Exception:
            return None

    @on(Button.Pressed, "#personas-lore-attach-add")
    def _attach_add(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(LoreAttachRequested())

    @on(Button.Pressed, "#personas-lore-attach-detach")
    def _attach_detach(self, event: Button.Pressed) -> None:
        event.stop()
        conversation_id = self._selected_attachment_id()
        if conversation_id is None:
            self.set_status("Select an attached conversation first.")
            return
        self.post_message(LoreDetachRequested(conversation_id))
```

- [ ] **Step 7: Update `__all__`**

In `__all__` (`:601`), add `"LoreAttachRequested",` and `"LoreDetachRequested",`.

- [ ] **Step 8: Run to verify they pass + no regressions**

Run the Step-2 command (PASS), then the whole file:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): Attachments tab in the Lore detail widget (I/O-free)"
```

---

### Task 4: Screen handlers + real-DB integration

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (import the messages + picker; add `_refresh_lore_attachments`; call it from `_select_lore_entry`; add attach/detach handlers)
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: `get_conversations_for_world_book` (Task 1); `ConversationAttachPicker` (Task 2); `LoreAttachRequested`/`LoreDetachRequested` (Task 3); existing `_lore_manager()`, `_list_attachable_conversations`, `associate_world_book_with_conversation`, `disassociate_world_book_from_conversation`, `self.state.selected_entity_id`, `self._io_dialog_active`, `self._notify`.

- [ ] **Step 1: Write the failing real-DB tests**

Append to `Tests/UI/test_personas_lore.py` (uses `LorePersonasTestApp` + `lore_db` + `seeded_lore_book`):
```python
@pytest.mark.asyncio
async def test_attach_via_picker_then_detach_real_db(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, monkeypatch
):
    from textual.widgets import TabbedContent
    from tldw_chatbook.Widgets.Persona_Widgets.conversation_attach_picker import (
        ConversationAttachPicker,
    )
    lore_db.add_conversation({"id": "conv-x", "title": "Noir case"})
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        screen.query_one("#personas-lore-tabs", TabbedContent).active = "personas-lore-tab-attachments"
        await pilot.pause()

        async def _fake_push(screen_obj):
            return "conv-x" if isinstance(screen_obj, ConversationAttachPicker) else None

        monkeypatch.setattr(screen.app, "push_screen_wait", _fake_push, raising=False)
        monkeypatch.setattr(
            screen, "_list_attachable_conversations",
            lambda: [{"conversation_id": "conv-x", "title": "Noir case"}],
        )
        screen.post_message(LoreAttachRequested())
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        attached = manager.get_conversations_for_world_book(seeded_lore_book["book_id"])
        assert [r["conversation_id"] for r in attached] == ["conv-x"]
        table = screen.query_one("#personas-lore-attachments-table", DataTable)
        assert table.row_count == 1
        # detach
        screen.post_message(LoreDetachRequested("conv-x"))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert manager.get_conversations_for_world_book(seeded_lore_book["book_id"]) == []


@pytest.mark.asyncio
async def test_attached_book_reaches_send_path_query(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    # Proves the (already-live) send path would inject an attached book:
    # get_world_books_for_conversation returns it after attach.
    lore_db.add_conversation({"id": "conv-y", "title": "Case Y"})
    manager = WorldBookManager(lore_db)
    manager.associate_world_book_with_conversation("conv-y", seeded_lore_book["book_id"])
    books = manager.get_world_books_for_conversation("conv-y", enabled_only=False)
    assert any(b["id"] == seeded_lore_book["book_id"] for b in books)
```
Add `LoreAttachRequested, LoreDetachRequested` to the test file's `personas_lore_detail` import (already added in Task 3).

- [ ] **Step 2: Run to verify they fail**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "attach_via_picker_then_detach or reaches_send_path" \
-q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — the attach test errors (no `@on(LoreAttachRequested)` handler → the message is unhandled, nothing persists). (`test_attached_book_reaches_send_path_query` may already PASS — it only uses the manager; that's fine, it's a live-path pin.)

- [ ] **Step 3: Import the messages + picker in the screen**

In `personas_screen.py`, add `LoreAttachRequested`, `LoreDetachRequested` to the existing `from ...Widgets.Persona_Widgets.personas_lore_detail import (...)` block, and add:
```python
from ...Widgets.Persona_Widgets.conversation_attach_picker import ConversationAttachPicker
```

- [ ] **Step 4: Add `_refresh_lore_attachments` + call it from `_select_lore_entry`**

Add the refresh method near the other lore handlers:
```python
    async def _refresh_lore_attachments(self) -> None:
        """Reload the Attachments tab for the selected lore book."""
        entity_id = self.state.selected_entity_id
        if self.state.selected_entity_kind != "lore" or not entity_id:
            return
        manager = self._lore_manager()
        if manager is None:
            return
        try:
            rows = await asyncio.to_thread(
                manager.get_conversations_for_world_book, int(entity_id)
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Could not load lore attachments.")
            self._notify(f"Could not load attachments: {exc}", "error")
            return
        try:
            self.query_one(PersonasLoreDetailWidget).load_attachments(rows)
        except QueryError:
            pass
```
In `_select_lore_entry`, after the book + entries are loaded into the detail widget (mirroring where the dictionary path calls `_refresh_dictionary_attachments`), add:
```python
        await self._refresh_lore_attachments()
```

- [ ] **Step 5: Add the attach + detach handlers**

Add near the other `@on(Lore...)` handlers:
```python
    @on(LoreAttachRequested)
    async def _handle_lore_attach(self, message: LoreAttachRequested) -> None:
        message.stop()
        if self.state.selected_entity_kind != "lore" or not self.state.selected_entity_id:
            return
        if self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._lore_attach_worker(), group="personas-io")

    async def _lore_attach_worker(self) -> None:
        try:
            entity_id = self.state.selected_entity_id
            manager = self._lore_manager()
            if manager is None or not entity_id:
                return
            convs = await asyncio.to_thread(self._list_attachable_conversations)
            try:
                picked = await self.app.push_screen_wait(ConversationAttachPicker(convs))
            except Exception:
                logger.opt(exception=True).warning("Could not show the attach picker.")
                return
            if not picked:
                return
            try:
                await asyncio.to_thread(
                    manager.associate_world_book_with_conversation,
                    str(picked),
                    int(entity_id),
                )
            except Exception as exc:
                logger.opt(exception=True).warning("Could not attach the world book.")
                self._notify(f"Attach failed: {exc}", "error")
                return
            await self._refresh_lore_attachments()
            self._notify("Attached to conversation.", "information")
        finally:
            self._io_dialog_active = False

    @on(LoreDetachRequested)
    async def _handle_lore_detach(self, message: LoreDetachRequested) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        if self.state.selected_entity_kind != "lore" or not entity_id:
            return
        manager = self._lore_manager()
        if manager is None:
            return
        try:
            await asyncio.to_thread(
                manager.disassociate_world_book_from_conversation,
                str(message.conversation_id),
                int(entity_id),
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Could not detach the world book.")
            self._notify(f"Detach failed: {exc}", "error")
            return
        await self._refresh_lore_attachments()
        self._notify("Detached from conversation.", "information")
```

- [ ] **Step 6: Run to verify they pass + full file**

Run the Step-2 command (PASS), then the whole file:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass.

- [ ] **Step 7: Full gate + commit**

```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_manager.py Tests/UI/test_conversation_attach_picker.py Tests/UI/test_personas_lore.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('APP IMPORT OK')"
```
Expected: all pass; `APP IMPORT OK`. Then:
```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): wire Attachments tab attach/detach to WorldBookManager"
```

---

## Notes for the reviewer

- **No migration, no send-path change:** the junction and `chat_events.py` application already exist. Any change to `chat_events.py` or a new migration is out of scope.
- **Attach is an upsert:** `associate_world_book_with_conversation` never raises `ConflictError`; there must be no conflict-handling branch (re-attach just updates priority).
- **String conversation ids:** every attach/detach/refresh path must pass `str(...)` conversation ids (the junction is `TEXT`).
- **I/O-free widget:** `load_attachments` takes rows; the widget does no DB. All DB access is in the screen, off-thread.
- **Reuse, don't duplicate:** the attach worker reuses `_list_attachable_conversations`; `DictionaryAttachPicker` is untouched.
- The `test_attached_book_reaches_send_path_query` test pins that an attached book is returned by `get_world_books_for_conversation` — the query the live legacy send path already uses.
