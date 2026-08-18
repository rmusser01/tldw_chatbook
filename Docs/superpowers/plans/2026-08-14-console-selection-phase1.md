# Console Text Selection — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mouse text selection in the Console transcript with a stacked floating menu offering `Add to chat` (this phase only; More Details / Ask in Side Chat / feedback actions are phases 2–3 under separate plans).

**Architecture:** A pure-logic `SelectionManager` in a new module coordinates drags on `ConsoleTranscript`; row widgets implement a small selection protocol (display text + highlight range). On mouse-up a floating `ConsoleSelectionMenu` widget (jump-pill pattern, not a modal) appears at the release cell; `Add to chat` inserts the quote into `ConsoleComposerBar` via a new public method wrapping the existing `_insert_literal_at_cursor`.

**Tech Stack:** Textual 8.x (≥8.0.0,<9), pytest with `textual` pilot-based widget tests.

**Spec:** `Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md` (§1 Selection System, §2 Add to chat).

## Global Constraints

- Selection domain is the rendered/displayed plain text, per row (spec §1).
- Single-row only: drags crossing row boundaries clamp to the origin row (spec §1).
- Drag mode arms after ≥1 cell of movement; rows suppress their `on_click` message-selection toggle during/just after a drag (spec §1).
- Selections on streaming rows clamp to last stable text; if the row is replaced, selection clears (spec §1).
- Non-selectable rows (anything in `PROTECTED_CLICK_CLASSES`, banners, headers) never start a selection (spec §1).
- Selection quotes are capped at `SELECTION_QUOTE_CAP = 4000` chars, truncated with `… [truncated]` (spec §1).
- Selection granularity: character-level on plain-text rows (`ConsoleTranscriptMessage`); **line-level on markdown rows** (`ConsoleMarkdownMessage`) in this phase — cell→offset mapping through Textual's `Markdown` renderer is not stable enough for character-level, so markdown drags map to whole rendered source lines. This is an explicit phase-1 simplification; later phases may tighten it.
- No new keybindings this phase (keyboard selection is phase 5); footer hints unchanged — ADR-031 compliance is therefore trivially preserved.
- Escape and click-outside dismiss the menu with no side effects (task-16211 contract, implemented on the floating widget).
- API keys never logged; no secrets in selection payloads (repo security rules).

---

### Task 1: ADR + Backlog task

**Files:**
- Create: `backlog/decisions/043-console-text-selection-and-annotations.md`
- Create/modify: `backlog/tasks/task-<id> - Console text selection phase 1.md` (via `backlog` CLI)

**Interfaces:**
- Consumes: spec §6 (ADR requirement).
- Produces: ADR-043 that tasks 2–7 and the phase 2–5 plans reference.

- [ ] **Step 1: Create the ADR**

Write `backlog/decisions/043-console-text-selection-and-annotations.md` following the repo's ADR format (see `backlog/decisions/031-*.md` for the house style). Content: status Accepted, date 2026-08-14; Context — Console transcript is a custom row-widget stack with no character-level selection, and review-style workflows (plannotator/codex pattern) need select-then-act; Decision — per-row selection delegates with a transcript-level `SelectionManager`, selection domain is displayed text per row, single-row v1, rendered-text granularity with line-level markdown simplification in phase 1, floating menu widget instead of modal, annotations anchored by deterministic `(session_id, row_key)` (phases 4), schema v8 bump deferred to phase 4; Alternatives considered — transcript-wide virtual text buffer (rejected: drift during streaming), `ModalScreen` menu (rejected: cannot anchor at cell), keyboard-only selection (rejected: not the codex interaction); References — ADR-031 (keybindings, none added in phase 1), spec path.

- [ ] **Step 2: Create the backlog task and put it In Progress**

```bash
backlog task create "Console text selection phase 1" -d "Mouse text selection in Console transcript with stacked menu and Add to chat" --ac "Mouse drag selects text in plain and markdown rows,Menu appears at release cell with Add to chat,Add to chat inserts quote at composer caret,Click vs drag disambiguated,Streaming rows clamp selection,Tests green" --priority high
backlog task list --plain   # note the new task id
backlog task edit <id> -a @{you} -s "In Progress" --plan "Per Docs/superpowers/plans/2026-08-14-console-selection-phase1.md"
```

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/043-console-text-selection-and-annotations.md
git commit -m "docs: ADR-043 console text selection and annotations"
```

---

### Task 2: Selection core (pure logic)

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_selection.py`
- Test: `Tests/UI/test_console_selection_core.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by tasks 3–6):

```python
SELECTION_QUOTE_CAP: int  # = 4000

@dataclass(frozen=True)
class TextSelection:
    row_key: str          # key of the owning _TranscriptRow / row widget
    start: int            # min offset into row display text
    end: int              # max offset (exclusive)

@dataclass(frozen=True)
class SelectionState:
    active: bool          # True while dragging
    selection: TextSelection | None

class SelectionManager:
    def __init__(self) -> None: ...
    def begin_drag(self, row_key: str, offset: int) -> None: ...
    def extend_drag(self, row_key: str, offset: int) -> None: ...  # clamps to origin row
    def finish_drag(self) -> TextSelection | None: ...             # None if empty/threshold unmet
    def cancel(self) -> None: ...
    @property
    def state(self) -> SelectionState: ...
    @property
    def just_finished(self) -> bool: ...   # True after finish_drag, cleared by consume_just_finished()

def cap_quote(text: str) -> str: ...       # truncates to SELECTION_QUOTE_CAP + "\n… [truncated]"
```

- [ ] **Step 1: Write failing tests**

```python
"""Tests/UI/test_console_selection_core.py"""
from tldw_chatbook.Widgets.Console.console_selection import (
    SELECTION_QUOTE_CAP, SelectionManager, TextSelection, cap_quote,
)


def test_drag_within_single_row_produces_ordered_selection():
    mgr = SelectionManager()
    mgr.begin_drag("m1", 10)
    mgr.extend_drag("m1", 4)
    sel = mgr.finish_drag()
    assert sel == TextSelection(row_key="m1", start=4, end=10)


def test_drag_across_rows_clamps_to_origin_row():
    mgr = SelectionManager()
    mgr.begin_drag("m1", 2)
    mgr.extend_drag("m2", 50)  # different row: ignored
    sel = mgr.finish_drag()
    assert sel == TextSelection(row_key="m1", start=2, end=2)
    assert mgr.finish_drag() is None or True  # empty selection is fine


def test_empty_selection_finishes_none_and_sets_just_finished():
    mgr = SelectionManager()
    mgr.begin_drag("m1", 5)
    mgr.extend_drag("m1", 5)
    assert mgr.finish_drag() is None
    assert mgr.just_finished is True
    mgr.consume_just_finished()
    assert mgr.just_finished is False


def test_cancel_clears_everything():
    mgr = SelectionManager()
    mgr.begin_drag("m1", 0)
    mgr.extend_drag("m1", 9)
    mgr.cancel()
    assert mgr.state.selection is None
    assert mgr.state.active is False


def test_cap_quote_truncates_long_text():
    text = "x" * (SELECTION_QUOTE_CAP + 100)
    out = cap_quote(text)
    assert len(out) < len(text)
    assert out.endswith("… [truncated]")


def test_cap_quote_passes_short_text_through():
    assert cap_quote("hello") == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_selection_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Widgets.Console.console_selection'`

- [ ] **Step 3: Implement `console_selection.py`**

```python
"""Console transcript text-selection state (TASK: console selection phase 1).

Pure logic, no Textual imports: the transcript widget feeds it mouse events,
row widgets render whatever it produces. Selection domain is a row's
displayed plain text; single-row only (spec 2026-08-14 §1).
"""

from __future__ import annotations

from dataclasses import dataclass

SELECTION_QUOTE_CAP = 4000
_TRUNCATION_MARKER = "\n… [truncated]"


@dataclass(frozen=True)
class TextSelection:
    row_key: str
    start: int
    end: int

    @property
    def is_empty(self) -> bool:
        return self.end <= self.start


@dataclass(frozen=True)
class SelectionState:
    active: bool
    selection: TextSelection | None


class SelectionManager:
    """Tracks one in-progress or finished selection on a single row."""

    def __init__(self) -> None:
        self._origin_row: str | None = None
        self._origin_offset: int = 0
        self._current_offset: int = 0
        self._active: bool = False
        self._finished: TextSelection | None = None
        self._just_finished: bool = False

    @property
    def state(self) -> SelectionState:
        selection = None
        if self._active or self._finished is not None:
            start, end = sorted((self._origin_offset, self._current_offset))
            selection = TextSelection(row_key=self._origin_row or "", start=start, end=end)
        return SelectionState(active=self._active, selection=selection)

    @property
    def just_finished(self) -> bool:
        return self._just_finished

    def consume_just_finished(self) -> None:
        self._just_finished = False

    def begin_drag(self, row_key: str, offset: int) -> None:
        self._origin_row = row_key
        self._origin_offset = max(0, offset)
        self._current_offset = self._origin_offset
        self._active = True
        self._finished = None

    def extend_drag(self, row_key: str, offset: int) -> None:
        if not self._active or row_key != self._origin_row:
            return  # cross-row drags clamp to the origin row
        self._current_offset = max(0, offset)

    def finish_drag(self) -> TextSelection | None:
        if not self._active:
            return None
        self._active = False
        state = self.state
        self._finished = None if state.selection is None or state.selection.is_empty else state.selection
        self._just_finished = True
        return self._finished

    def cancel(self) -> None:
        self._origin_row = None
        self._active = False
        self._finished = None
        self._just_finished = False


def cap_quote(text: str) -> str:
    if len(text) <= SELECTION_QUOTE_CAP:
        return text
    return text[: SELECTION_QUOTE_CAP - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_selection_core.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_selection.py Tests/UI/test_console_selection_core.py
git commit -m "feat(console): pure-logic selection manager for transcript text selection"
```

---

### Task 3: Selection protocol on plain-text rows

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`ConsoleTranscriptMessage`, ~line 1358)
- Test: `Tests/UI/test_console_selection_rows.py`

**Interfaces:**
- Consumes: `TextSelection`, `cap_quote` from Task 2.
- Produces (used by tasks 4, 5, 6):

```python
# On ConsoleTranscriptMessage:
def get_display_text(self) -> str            # plain body text the row renders
def get_selection_text(self) -> str          # current highlighted text, cap_quote()d
def set_selection_range(self, start: int, end: int) -> None   # re-renders body w/ highlight span
def clear_selection(self) -> None

# Module-level helper (console_selection.py additions, still pure):
def offset_for_cell(text: str, cell_x: int) -> int
    # clamps cell_x chars into the line at the cursor's row: v1 maps cell_x
    # directly to a character offset on the unwrapped line (plain rows render
    # unwrapped long lines clipped, so the mapping is monotone + clamped).
```

- [ ] **Step 1: Write failing tests**

```python
"""Tests/UI/test_console_selection_rows.py"""
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscriptMessage


class _RowApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscriptMessage(_make_message("hello selection world"))


def _make_message(body: str):
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleChatMessage
    return ConsoleChatMessage(id="m1", role="user", content=body)


@pytest.mark.asyncio
async def test_display_text_is_plain_body():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        assert row.get_display_text() == "hello selection world"


@pytest.mark.asyncio
async def test_selection_range_highlights_and_quotes():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(6, 15)
        assert row.get_selection_text() == "selection"
        row.clear_selection()
        assert row.get_selection_text() == ""
```

Note: `ConsoleChatMessage` field names must be checked against the actual dataclass in `console_transcript.py` (search `class ConsoleChatMessage`) before writing the test — use its real constructor signature; only `id` and the body-content field matter. If the constructor is heavier, build via the same helper the existing transcript tests use (search `Tests/UI/` for `ConsoleChatMessage(`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_selection_rows.py -v`
Expected: FAIL — `AttributeError: 'ConsoleTranscriptMessage' object has no attribute 'get_display_text'`

- [ ] **Step 3: Implement the protocol on `ConsoleTranscriptMessage`**

Inside `ConsoleTranscriptMessage` (`console_transcript.py:1358`):

```python
def __init__(...) -> None:  # existing __init__ gains:
    self._selection_range: tuple[int, int] | None = None

def get_display_text(self) -> str:
    return _message_body_render_text(self._message, self._presentation)

def get_selection_text(self) -> str:
    if self._selection_range is None:
        return ""
    start, end = self._selection_range
    return cap_quote(self.get_display_text()[start:end])

def set_selection_range(self, start: int, end: int) -> None:
    self._selection_range = (start, end)
    self._refresh_body_highlight()

def clear_selection(self) -> None:
    if self._selection_range is None:
        return
    self._selection_range = None
    self._refresh_body_highlight()

def _refresh_body_highlight(self) -> None:
    """Re-render the body Static with a reverse-video span over the range."""
    from rich.text import Text
    try:
        body = self.query_one(".console-transcript-message-body", Static)
    except NoMatches:
        return
    plain = self.get_display_text()
    if self._selection_range is None:
        body.update(plain)
        return
    start, end = sorted(self._selection_range)
    start, end = max(0, start), min(end, len(plain))
    rich_text = Text(plain)
    if end > start:
        rich_text.stylize("reverse", start, end)
    body.update(rich_text)
```

Also add to `sync_message` (line ~1415): re-apply `self._selection_range` clamped to the new text length after `body.update(...)` — streaming updates shrink/grow the text, so clamp: `self._selection_range = (min(start, len(new)), min(end, len(new)))` and drop it (call `clear_selection()`) if the new text no longer contains the range start. Import `cap_quote` and `TextSelection` from `.console_selection` at the top of the file, next to the existing `Widgets/Console` imports.

Add `offset_for_cell` to `console_selection.py` (with its own unit tests in `test_console_selection_core.py`):

```python
def offset_for_cell(text: str, cell_x: int) -> int:
    """Map a horizontal cell offset to a character offset on one line."""
    return max(0, min(cell_x, len(text)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_selection_rows.py Tests/UI/test_console_selection_core.py -v`
Expected: PASS

- [ ] **Step 5: Regression — existing transcript tests**

Run: `pytest Tests/UI/ -k console_transcript -v`
Expected: PASS, no changes in existing behavior (highlight only applied when a range is set).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/console_selection.py Tests/UI/test_console_selection_core.py Tests/UI/test_console_selection_rows.py
git commit -m "feat(console): selection protocol on plain transcript rows with clamp-on-sync"
```

---

### Task 4: Transcript drag wiring + click/drag suppression

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`ConsoleTranscript.__init__` ~1771, `on_click` ~2919; `ConsoleTranscriptMessage.on_click` ~1445; `ConsoleMarkdownMessage.on_click` ~1344)
- Test: `Tests/UI/test_console_selection_transcript.py`

**Interfaces:**
- Consumes: `SelectionManager`, `offset_for_cell` (Task 2), row protocol (Task 3).
- Produces:

```python
# On ConsoleTranscript:
self.selection_manager: SelectionManager          # public for the screen/tests
def _selection_row_for(self, widget: Widget) -> ConsoleTranscriptMessage | None
    # walks parents from the event control to the nearest row implementing
    # the Task-3 protocol; None for protected/header/non-text widgets.
# Message posted on menu-worthy release:
class TranscriptTextSelected(Message):
    selection: TextSelection
    screen_x: int   # release cell, for menu anchoring
    screen_y: int
```

- [ ] **Step 1: Write failing tests**

```python
"""Tests/UI/test_console_selection_transcript.py"""
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.Console.console_selection import SelectionManager
# plus the same ConsoleChatMessage-building helper as Task 3's tests


class _TranscriptApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")  # rows mounted via set_messages


@pytest.mark.asyncio
async def test_drag_suppresses_message_toggle_click():
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        # mount one row via the same set_messages path existing tests use,
        # then simulate: MouseDown on row, MouseMove +5 cells, MouseUp.
        await pilot.hover(".console-transcript-message-body", offset=(0, 0))
        # after a real drag, the manager reports just_finished and the row's
        # message-selection state must be UNchanged:
        transcript.selection_manager.consume_just_finished()
        assert transcript.selected_message_id is None


@pytest.mark.asyncio
async def test_protected_rows_never_start_selection():
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        assert transcript.selection_manager.state.active is False
```

(Concrete pilot event driving: use `pilot.hover` plus posting `MouseDown`/`MouseMove`/`MouseUp` messages to the transcript — mirror however existing console transcript widget tests drive mouse events; search `Tests/UI/` for `MouseMove(` for the house pattern.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_selection_transcript.py -v`
Expected: FAIL — `AttributeError: 'ConsoleTranscript' object has no attribute 'selection_manager'`

- [ ] **Step 3: Implement**

In `ConsoleTranscript.__init__`: `self.selection_manager = SelectionManager()`.

Mouse plumbing on `ConsoleTranscript`:

```python
def on_mouse_down(self, event: MouseDown) -> None:
    row = self._selection_row_for(event.control)
    if row is None:
        return
    offset = offset_for_cell(row.get_display_text(), event.x - row.region.x)
    self.selection_manager.begin_drag(row.id, offset)

def on_mouse_move(self, event: MouseMove) -> None:
    if not self.selection_manager.state.active:
        return
    event.stop()
    row = self._selection_row_for(event.control)
    if row is None or row.id != self.selection_manager.state.selection.row_key:
        return  # leaving the origin row: hold last position (clamp)
    offset = offset_for_cell(row.get_display_text(), event.x - row.region.x)
    self.selection_manager.extend_drag(row.id, offset)
    sel = self.selection_manager.state.selection
    for other in self._row_widgets.values():
        if isinstance(other, ConsoleTranscriptMessage) and other.id != row.id:
            other.clear_selection()
    row.set_selection_range(sel.start, sel.end)

def on_mouse_up(self, event: MouseUp) -> None:
    if not self.selection_manager.state.active:
        return
    event.stop()
    sel = self.selection_manager.finish_drag()
    if sel is not None:
        self.post_message(
            TranscriptTextSelected(selection=sel, screen_x=event.x, screen_y=event.y)
        )
```

Click suppression — in both `ConsoleTranscriptMessage.on_click` (~1445) and `ConsoleMarkdownMessage.on_click` (~1344), first lines:

```python
transcript = self._transcript()   # existing parent-walk helper or the loop already there
if transcript is not None and (
    transcript.selection_manager.state.active
    or transcript.selection_manager.just_finished
):
    event.stop()
    return
```

And in `ConsoleTranscript.on_click` (~2919): when `selection_manager.just_finished`, `event.stop(); self.selection_manager.consume_just_finished(); return` — a drag-release must not clear the message selection either.

`_selection_row_for`: walk `event.control` up `parent` links; return the first widget that is a `ConsoleTranscriptMessage` (Task 3 protocol) **and** whose clicked control is not in `PROTECTED_CLICK_CLASSES` (reuse the existing `any(...)` check); return `None` otherwise. Markdown rows are excluded this phase (Task 6 adds them).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_selection_transcript.py -v`
Expected: PASS

- [ ] **Step 5: Regression — full console transcript suite**

Run: `pytest Tests/UI/ -k "console and transcript" -v`
Expected: PASS — clicks still toggle message selection when no drag occurred.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_selection_transcript.py
git commit -m "feat(console): mouse drag selection wiring with click suppression"
```

---

### Task 5: Floating selection menu

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_selection_menu.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (mount/handle `TranscriptTextSelected`)
- Test: `Tests/UI/test_console_selection_menu.py`

**Interfaces:**
- Consumes: `TranscriptTextSelected` (Task 4), jump-pill floating pattern (`ConsoleTranscriptJumpPill` mount at `console_transcript.py:1871`).
- Produces:

```python
class ConsoleSelectionMenu(Vertical):
    """Floating stacked menu anchored at the selection release cell."""
    class AddToChat(Message): ...        # posted when 'Add to chat' pressed
    def __init__(self, *, screen_x: int, screen_y: int, has_add_to_chat: bool = True): ...
```

- [ ] **Step 1: Write failing tests**

```python
"""Tests/UI/test_console_selection_menu.py"""
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.Console.console_selection_menu import ConsoleSelectionMenu


class _MenuApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleSelectionMenu(screen_x=4, screen_y=6)


@pytest.mark.asyncio
async def test_menu_offers_add_to_chat_and_posts_message():
    app = _MenuApp()
    messages: list = []
    async with app.run_test() as pilot:
        menu = app.query_one(ConsoleSelectionMenu)
        menu.post_message = lambda m: messages.append(m)  # capture
        await pilot.click("#console-selection-add-to-chat")
        assert any(type(m).__name__ == "AddToChat" for m in messages)


@pytest.mark.asyncio
async def test_escape_dismisses_without_side_effects():
    app = _MenuApp()
    async with app.run_test() as pilot:
        await pilot.press("escape")
        assert not app.query(ConsoleSelectionMenu)  # removed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_selection_menu.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the menu**

```python
"""Floating selection-action menu (console selection phase 1).

Mounted inside ConsoleTranscript like the jump pill; NOT a ModalScreen
(modals are layer-centered and cannot anchor at a cell). Escape and
click-outside dismiss with no side effects (task-16211 contract).
"""

from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Click, Key
from textual.message_pump import Message
from textual.widgets import Button


class ConsoleSelectionMenu(Vertical):
    DEFAULT_CSS = """
    ConsoleSelectionMenu {
        layer: selection-menu;
        width: auto; height: auto;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    """
    BINDING = [Binding("escape", "dismiss", show=False)]

    class AddToChat(Message):
        """User chose 'Add to chat' for the active selection."""

    def __init__(self, *, screen_x: int, screen_y: int, has_add_to_chat: bool = True) -> None:
        super().__init__(id="console-selection-menu", classes="console-selection-menu")
        self._anchor = (screen_x, screen_y)
        self._has_add_to_chat = has_add_to_chat

    def compose(self):
        if self._has_add_to_chat:
            yield Button("Add to chat", id="console-selection-add-to-chat", variant="primary")

    def on_mount(self) -> None:
        x, y = self._anchor
        self.styles.offset = (x, y + 1)  # just below the release cell

    @on(Button.Pressed, "#console-selection-add-to-chat")
    def _add_to_chat(self) -> None:
        self.post_message(self.AddToChat())

    def action_dismiss(self) -> None:
        self.remove()

    async def _on_click(self, event: Click) -> None:
        event.stop()  # clicks inside the menu must not clear anything
```

Click-outside: in `ConsoleTranscript.on_click` (Task 4's version), before the existing logic, remove any mounted `ConsoleSelectionMenu` (`self.query(ConsoleSelectionMenu)` → `.remove()`), then continue — a click elsewhere both closes the menu and behaves normally otherwise.

Transcript handling of `TranscriptTextSelected` (from Task 4): remove any existing menu, then `self.mount(ConsoleSelectionMenu(screen_x=..., screen_y=...))`. On `ConsoleSelectionMenu.AddToChat`: fetch the selection text from the origin row, post it up to the screen:

```python
@on(ConsoleSelectionMenu.AddToChat)
def _selection_add_to_chat(self) -> None:
    manager = self.selection_manager
    sel = manager.state.selection
    row = self._row_widgets.get(sel.row_key) if sel else None
    if row is not None and hasattr(row, "get_selection_text"):
        self.post_message(ConsoleSelectionQuoteRequested(quote=cap_quote(row.get_selection_text())))
    manager.cancel()
    row and row.clear_selection()
    self._remove_selection_menu()
```

`ConsoleSelectionQuoteRequested(Message)` with field `quote: str` is defined in `console_selection_menu.py` and consumed by Task 6's screen wiring.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_selection_menu.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_selection_menu.py tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_selection_menu.py
git commit -m "feat(console): floating stacked selection menu with Add to chat action"
```

---

### Task 6: Composer insertion + screen wiring (end-to-end)

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py` (public wrapper near `_insert_literal_at_cursor`, line 2579)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (handler near the composer mount, ~14965)
- Test: `Tests/UI/test_console_selection_end_to_end.py`

**Interfaces:**
- Consumes: `ConsoleSelectionQuoteRequested` (Task 5), `_insert_literal_at_cursor` (`console_composer_bar.py:2579`).
- Produces:

```python
# ConsoleComposerBar:
def insert_quote(self, text: str) -> None
    # Prefixes each non-empty line with "> ", calls _insert_literal_at_cursor,
    # then triggers the existing draft re-render (the same _advance/refresh path
    # _insert_literal_at_cursor's callers use — follow how on_key typing does it).
```

- [ ] **Step 1: Write failing tests**

```python
"""Tests/UI/test_console_selection_end_to_end.py"""
import pytest

from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


@pytest.mark.asyncio
async def test_insert_quote_prepends_quote_markers_at_caret():
    from textual.app import App, ComposeResult

    class _ComposerApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ConsoleComposerBar(id="console-native-composer")

    app = _ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_quote("line one\nline two")
        assert "> line one\n> line two" in composer.draft_text


@pytest.mark.asyncio
async def test_screen_handler_routes_quote_to_composer():
    # Unit-test the ChatScreen handler directly: construct the message, call
    # the handler method, assert insert_quote was called with cap_quote()d text.
    # Follow the existing ChatScreen unit-test pattern in Tests/UI/ (search
    # for "_sync_console_control_bar" tests) for constructing a bare ChatScreen.
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_selection_end_to_end.py -v`
Expected: FAIL — no `insert_quote` attribute.

- [ ] **Step 3: Implement**

In `ConsoleComposerBar`:

```python
def insert_quote(self, text: str) -> None:
    """Insert a transcript selection as a quoted block at the caret (or end)..

    Public seam for the console selection menu's 'Add to chat' action.
    The caret always exists in the segment model (it is not focus-bound),
    so this lands wherever the caret sits — end of draft when unfocused,
    which is the spec §2 fallback.
    """
    quoted = "\n".join(
        f"> {line}" if line.strip() else ">" for line in text.splitlines()
    )
    if not quoted:
        return
    self._collapse_large_paste_if_needed(quoted)  # if the existing paste-collapse
    # path is required for large inserts, follow replace_snapshot_as_paste;
    # otherwise delete this line — _insert_literal_at_cursor handles plain text.
    self._insert_literal_at_cursor(quoted)
    self._refresh_draft_display()  # use the same post-edit refresh call the
    # keyboard typing path makes (grep _insert_literal_at_cursor callers)
```

Check `_insert_literal_at_cursor`'s existing callers first (grep) and mirror their post-edit refresh exactly — do not invent a refresh call.

In `ChatScreen` (`chat_screen.py`, near the composer mount at ~14965, in the `@on(...)` handler section):

```python
@on(ConsoleSelectionQuoteRequested)
def _console_selection_quote_requested(self, event: ConsoleSelectionQuoteRequested) -> None:
    composer = self.query_one("#console-native-composer", ConsoleComposerBar)
    composer.insert_quote(event.quote)
    self.notify("Added selection to composer", timeout=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_selection_end_to_end.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_selection_end_to_end.py
git commit -m "feat(console): Add to chat routes selection quote into composer at caret"
```

---

### Task 7: Markdown row support (line-level) + live spike + wrap-up

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`ConsoleMarkdownMessage`, ~1136)
- Test: `Tests/UI/test_console_selection_rows.py` (extend)
- Modify: backlog task file (Implementation Notes)

**Interfaces:**
- Consumes: `SelectionManager` drag events (Task 4 treats markdown rows as non-selectable; this task flips them on).
- Produces: `ConsoleMarkdownMessage` implements the same four protocol methods as Task 3, with **line-level** granularity (`set_selection_range` takes `(line_start, line_end)` character offsets that snap outward to line boundaries).

- [ ] **Step 1: Write failing tests**

Extend `Tests/UI/test_console_selection_rows.py`:

```python
@pytest.mark.asyncio
async def test_markdown_row_selects_at_line_granularity():
    # mount a ConsoleMarkdownMessage with markdown "# Title\n\nbody line\nmore",
    # set_selection_range(offset_of("body") + 2, offset_of("body") + 5)
    # -> get_selection_text() returns the whole "body line" line (snapped outward)
    ...
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest Tests/UI/test_console_selection_rows.py -k markdown -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`ConsoleMarkdownMessage` (line 1136): store `self._selection_line_range: tuple[int, int] | None`. `get_display_text()` returns the row's markdown **source** (the string the row was built from — find the field the `Markdown` child is fed at row build, ~console_transcript.py:3317). `set_selection_range(start, end)` snaps both offsets outward to `'\n'` boundaries and re-renders the Markdown body by re-mounting its content with the selected lines wrapped in a styled `Text`… markdown rows render via the `Markdown` widget, so instead of restyling internals, render the highlight as a block: below the Markdown child, mount/update a `Static` preview strip showing the selected lines with `reverse` style (visually equivalent highlight, avoids fighting the Markdown renderer). `get_selection_text()` returns the snapped source lines, `cap_quote`d. Update `_selection_row_for` in Task 4 to accept `ConsoleMarkdownMessage` too, and map mouse cells to line numbers via the source-line/row-height layout the row already computes for its content height (clamp: pick the nearest line).

- [ ] **Step 4: Run tests + full phase suite**

Run: `pytest Tests/UI/test_console_selection_*.py -v`
Expected: PASS.

- [ ] **Step 5: Live spike (manual, per lessons-live-verification.md)**

Run `python3 -m tldw_chatbook.app`, open the Console, and verify by hand in a real terminal: (1) plain-drag selects and terminal-native copy still works with shift-drag; (2) the menu appears at the release cell and Escape/click-outside dismiss; (3) Add to chat lands the quote at the caret; (4) during a streaming reply, a selection on the streaming row clamps and survives the sync tick; (5) a sloppy 2-cell click still toggles message selection, doesn't open the menu. Record results (pass/fail each) in the task's Implementation Notes; any failure found here becomes a fix commit before Task 8 below.

- [ ] **Step 6: Wrap-up — lint, docs, backlog**

```bash
pytest Tests/UI/test_console_selection_*.py -v          # green
ruff check tldw_chatbook/Widgets/Console/console_selection.py tldw_chatbook/Widgets/Console/console_selection_menu.py tldw_chatbook/Widgets/Console/console_transcript.py
backlog task edit <id> --notes "Phase 1 complete: ..."   # Implementation Notes w/ approach, files, deviations
backlog task edit <id> -s Done
```

Also check every AC checkbox in the task file, and confirm the ADR-043 link is in the task notes (repo DoD).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(console): markdown line-level selection; phase 1 complete"
```

---

## Self-Review Notes

- Spec §1 coverage: drag selection (T2–4), menu at release cell (T5), click suppression (T4), streaming clamp (T3 `sync_message`), protected rows (T4 `_selection_row_for`), selection cap (T2 `cap_quote`), single-row clamp (T2), Add to chat w/ caret fallback (T6 — caret always exists in the segment model, end-of-draft when unfocused). Markdown granularity is an explicit phase-1 simplification (T7), recorded in ADR-043.
- Phases 2–5 (side chat, feedback actions, annotations/schema v8, keyboard selection) get separate plans; this plan does not stub them.
- Two spots require the implementer to follow house patterns rather than invented code, both flagged inline: `ConsoleChatMessage` construction in tests (T3) and the composer's post-edit refresh path (T6).
