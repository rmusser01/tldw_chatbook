# Personas Workbench (Characters + Personas) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Character/Persona creation, viewing, editing, and management surface as a destination-native workbench on the `personas` route, replacing the thin snapshot shell and the legacy `ccp` route.

**Architecture:** New `PersonasScreen` workbench shell (header → mode strip → library | work area | inspector → footer shortcuts) with new small pane widgets in `Widgets/Persona_Widgets/`. The existing CCP behavior layer is reused: `CCPCharacterHandler`/`CCPPersonaHandler`, `CCPCharacterCardWidget`/`CCPCharacterEditorWidget`, `Character_Chat_Lib` import/export/validation, `CharacterPersonaScopeService`. Legacy `ccp_screen.py` and sidebar-era chrome retire in the final phase after the route flip.

**Tech Stack:** Python 3.11, Textual screens/widgets, TCSS design-system classes (`ds-*`), pytest + `run_test()` mounted tests, real in-memory SQLite where DB behavior matters.

**Spec:** `Docs/superpowers/specs/2026-06-09-personas-workbench-design.md`

**ADR required:** yes
**ADR path:** `backlog/decisions/007-personas-workbench-route-consolidation.md` (landed via PR #506)
**Reason:** Route consolidation, module retirement, and long-lived UX structure.

## Revision 2026-06-10: PR #506 adoption

PR #506 ("Add Personas workbench foundation contracts") landed the foundation
independently. Tasks 1 and 2 are SUPERSEDED — do not execute them as written:

- The ADR is `backlog/decisions/007-personas-workbench-route-consolidation.md`
  (ADR-007), not ADR-004. The backlog task is task-90.
- `Widgets/Persona_Widgets/personas_state.py` and `personas_messages.py` exist with a
  different API than Tasks 1-2 describe. Keep these two files (and
  `Persona_Widgets/__init__.py`) byte-identical to PR #506 until it merges; new message
  classes needed by later tasks go in a NEW sibling module
  `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py` (created by the
  first task that needs it, extended additively after that).

Substitutions for all later tasks:

| Plan as written | Use instead |
| --- | --- |
| `LibraryRowSelected(kind, item_id)` | `PersonaEntitySelected(entity_kind=..., entity_id=..., entity_name=...)` |
| `LibrarySearchChanged(query)` | `PersonaSearchChanged(query=...)` |
| `LibraryNewRequested()` | `PersonaActionRequested(action="create")` |
| `LibraryImportRequested()` | `PersonaActionRequested(action="import")` |
| `PersonasWorkbenchState` frozen + `replace()` / `with_mode()` | mutable instance: `state.switch_mode(mode)`, `state.select_entity(...)`, `state.clear_selection()` |
| `state.selected_kind` / `state.selected_id` | `state.selected_entity_kind` / `state.selected_entity_id` |
| `state.is_unsaved` | `state.has_unsaved_changes` |
| `state.edit_mode` | screen attribute `self._edit_mode` ("view"/"edit"/"create") — not part of shared state |
| `PERSONAS_MODES` / `PLACEHOLDER_MODES` | `VALID_PERSONA_MODES` from `personas_state` (includes `import_export`, which gets NO mode chip — Import/Export stay toolbar actions per spec); placeholder modes are `("prompts", "dictionaries", "lore")` defined screen-side |
| `MODE_LABELS` defined in screen | import `MODE_LABELS` from `personas_state` |

Messages still to be added (in the task that first needs them, additively):
`ConversationRowSelected`, `PreviewReplyRequested`, `PreviewResetRequested`,
`PreviewOpenInConsoleRequested`, `EditPersonaRequested`,
`PersonaProfileSaveRequested`, `PersonaProfileEditCancelled` — constructors exactly as
written in Tasks 2/7/11/13. `Tests/UI/test_personas_workbench_state.py` from Task 2 is
NOT created; PR #506's `Tests/UI/test_personas_workbench_foundation.py` covers the
state model.

**Conventions for every task:**
- Run tests with `python -m pytest -q <path> --tb=short` (no `timeout` command in this environment).
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Status text must be readable labels; tests assert IDs/classes/text, never colors.

---

## Phase 1 — Workbench shell on the `personas` route, Characters mode

### Task 1: ADR and backlog task

**Files:**
- Create: `backlog/decisions/004-personas-destination-native-workbench.md`

- [ ] **Step 1: Create the backlog task**

```bash
backlog task create "Personas destination-native workbench (Characters + Personas)" \
  -d "Rebuild the Character/Persona management surface as a destination-native workbench on the personas route per Docs/superpowers/specs/2026-06-09-personas-workbench-design.md" \
  -s "In Progress" \
  --ac "Personas route renders the destination workbench with Characters and Personas modes,Character create/view/edit/import/export/search work,Persona profile create/view/edit work,Saved conversations for a character are viewable,Preview conversation works without persisting,Attach to Console and Start Chat stage correct handoffs,Legacy ccp route resolves to the personas destination,Legacy CCP screen and sidebar modules are removed"
```

Note the created task id (referred to as `task-N` below).

- [ ] **Step 2: Write the ADR**

Create `backlog/decisions/004-personas-destination-native-workbench.md`:

```markdown
# ADR-004: Personas destination-native workbench and CCP route retirement

Status: Accepted
Date: 2026-06-09
Related Task: backlog/tasks/task-N - Personas destination-native workbench.md
Supersedes: N/A

## Decision

The `personas` route owns a single destination-native workbench for Characters and
Personas (create/view/edit/manage, import/export, preview, Console attachment). The
legacy `ccp` route and its screen (`ccp_screen.py`), the `conversation_screen.py`
re-export shim, and sidebar-era chrome (`ccp_sidebar_widget.py`,
`ccp_sidebar_handler.py`) are retired; `ccp`, `characters`, and `prompts` legacy
routes resolve to `personas`.

## Context

The Personas destination was split across a thin snapshot shell (`personas` route)
and a half-converted legacy workbench (`ccp` route), forcing two-hop navigation and
duplicating attachment logic. Console, Library, and Notes already follow the
destination workbench grammar defined in
`Docs/Design/agentic-terminal-visual-system.md`. The CCP behavior layer (handlers,
character card/editor widgets, import/validation libraries, scope service) is sound
and is reused; only the route shell, persona stubs, and sidebar chrome are replaced.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Finish converting `ccp_screen.py` in place, flip routes last | Keeps building on an 1,800-line screen with sidebar-era state; two parallel screens persist; hardest path to Notes-level layout quality |
| Clean rebuild including the data/behavior layer | Re-implements an 866-line character editor and battle-tested import flows for no product gain; highest regression risk |
| Separate Characters and Personas top-nav destinations | Violates the top-navigation contract (stable global destinations); New_UI images are layout references, not nav requirements |

## Consequences

- One management surface; Library keeps ownership of full conversation browsing.
- New pane widgets live in `Widgets/Persona_Widgets/`; reused CCP widgets keep their
  import paths and internal IDs (`#ccp-character-card-view`, `#ccp-character-editor-view`).
- Selection/search/preview message classes move out of the retired screen into
  `Widgets/Persona_Widgets/personas_messages.py`.
- Server-backed CRUD is out of scope; authority labels keep the seam visible.

## Links

- Spec: Docs/superpowers/specs/2026-06-09-personas-workbench-design.md
- Plan: Docs/superpowers/plans/2026-06-09-personas-workbench-implementation.md
```

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/004-personas-destination-native-workbench.md backlog/tasks/
git commit -m "docs: add ADR-004 Personas destination-native workbench

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 2: Workbench state and message vocabulary

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/__init__.py`
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_state.py`
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`
- Test: `Tests/UI/test_personas_workbench_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_personas_workbench_state.py
"""Unit tests for the Personas workbench state model."""

from dataclasses import replace

import pytest

from tldw_chatbook.Widgets.Persona_Widgets.personas_state import (
    PERSONAS_MODES,
    PLACEHOLDER_MODES,
    PersonasWorkbenchState,
)


def test_default_state_is_characters_view_mode():
    state = PersonasWorkbenchState()
    assert state.active_mode == "characters"
    assert state.edit_mode == "view"
    assert state.selected_kind is None
    assert state.selected_id is None
    assert state.is_unsaved is False
    assert state.search_query == ""


def test_mode_vocabulary_matches_contract():
    assert PERSONAS_MODES == ("characters", "personas", "prompts", "dictionaries", "lore")
    assert PLACEHOLDER_MODES == frozenset({"prompts", "dictionaries", "lore"})


def test_with_mode_clears_selection_and_edit_state():
    state = PersonasWorkbenchState(
        selected_kind="character",
        selected_id="3",
        edit_mode="edit",
        is_unsaved=True,
        search_query="sam",
    )
    switched = state.with_mode("personas")
    assert switched.active_mode == "personas"
    assert switched.selected_kind is None
    assert switched.selected_id is None
    assert switched.edit_mode == "view"
    assert switched.is_unsaved is False
    assert switched.search_query == ""


def test_with_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        PersonasWorkbenchState().with_mode("flashcards")


def test_state_is_frozen():
    state = PersonasWorkbenchState()
    with pytest.raises(Exception):
        state.active_mode = "personas"
    assert replace(state, active_mode="personas").active_mode == "personas"
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q Tests/UI/test_personas_workbench_state.py --tb=short`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Widgets.Persona_Widgets'`

- [ ] **Step 3: Implement state and messages**

```python
# tldw_chatbook/Widgets/Persona_Widgets/personas_state.py
"""State model for the Personas destination workbench."""

from __future__ import annotations

from dataclasses import dataclass, replace

PERSONAS_MODES: tuple[str, ...] = ("characters", "personas", "prompts", "dictionaries", "lore")
PLACEHOLDER_MODES: frozenset[str] = frozenset({"prompts", "dictionaries", "lore"})


@dataclass(frozen=True)
class PersonasWorkbenchState:
    """Selection, edit, and search state for the workbench panes."""

    active_mode: str = "characters"
    selected_kind: str | None = None  # "character" | "persona_profile"
    selected_id: str | None = None
    edit_mode: str = "view"  # "view" | "edit" | "create"
    is_unsaved: bool = False
    search_query: str = ""

    def with_mode(self, mode: str) -> "PersonasWorkbenchState":
        """Switch modes, clearing selection, edit, and search state."""
        if mode not in PERSONAS_MODES:
            raise ValueError(f"Unknown Personas workbench mode: {mode}")
        return replace(
            self,
            active_mode=mode,
            selected_kind=None,
            selected_id=None,
            edit_mode="view",
            is_unsaved=False,
            search_query="",
        )
```

```python
# tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py
"""Messages posted by Personas workbench panes.

Defined here (not in any screen module) because the legacy ccp_screen.py that
hosted equivalents is retired in Phase 4 of the workbench plan.
"""

from __future__ import annotations

from typing import Any, Dict

from textual.message import Message


class PersonasWorkbenchMessage(Message):
    """Base class for Personas workbench messages."""


class LibraryRowSelected(PersonasWorkbenchMessage):
    def __init__(self, kind: str, item_id: str) -> None:
        self.kind = kind  # "character" | "persona_profile"
        self.item_id = item_id
        super().__init__()


class LibrarySearchChanged(PersonasWorkbenchMessage):
    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__()


class LibraryNewRequested(PersonasWorkbenchMessage):
    pass


class LibraryImportRequested(PersonasWorkbenchMessage):
    pass


class ConversationRowSelected(PersonasWorkbenchMessage):
    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        super().__init__()


class PreviewReplyRequested(PersonasWorkbenchMessage):
    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__()


class PreviewResetRequested(PersonasWorkbenchMessage):
    pass


class PreviewOpenInConsoleRequested(PersonasWorkbenchMessage):
    pass


class EditPersonaRequested(PersonasWorkbenchMessage):
    def __init__(self, persona_id: str) -> None:
        self.persona_id = persona_id
        super().__init__()


class PersonaProfileSaveRequested(PersonasWorkbenchMessage):
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data
        super().__init__()


class PersonaProfileEditCancelled(PersonasWorkbenchMessage):
    pass
```

```python
# tldw_chatbook/Widgets/Persona_Widgets/__init__.py
"""Destination-native Personas workbench pane widgets."""
```

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench_state.py --tb=short`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/ Tests/UI/test_personas_workbench_state.py
git commit -m "feat: add Personas workbench state and message vocabulary

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: Library pane widget

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- Test: `Tests/UI/test_personas_library_pane.py`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_personas_library_pane.py
"""Mounted tests for the Personas library pane."""

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    LibraryRow,
    PersonasLibraryPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    LibraryNewRequested,
    LibraryRowSelected,
    LibrarySearchChanged,
)

pytestmark = pytest.mark.asyncio


class LibraryPaneApp(App):
    def __init__(self):
        super().__init__()
        self.received = []

    def compose(self):
        yield PersonasLibraryPane(id="personas-library-pane")

    def on_personas_library_pane_message(self, message) -> None:  # pragma: no cover
        pass


async def test_pane_renders_search_toolbar_and_empty_state():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        assert pilot.app.query_one("#personas-library-search", Input)
        assert pilot.app.query_one("#personas-library-new", Button)
        assert pilot.app.query_one("#personas-library-import", Button)
        pane.update_rows((), total=0, noun="characters")
        await pilot.pause()
        empty = pilot.app.query_one("#personas-library-empty", Static)
        assert "No characters yet" in str(empty.renderable)


async def test_update_rows_renders_rows_and_count():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        rows = (
            LibraryRow(item_id="1", kind="character", name="Detective Sam"),
            LibraryRow(item_id="2", kind="character", name="Tutor", is_unsaved=True),
        )
        pane.update_rows(rows, total=2, noun="characters")
        await pilot.pause()
        buttons = pilot.app.query(".personas-library-row")
        assert len(buttons) == 2
        assert "is-unsaved" in pilot.app.query_one(
            "#personas-library-row-character-2", Button
        ).classes
        count = pilot.app.query_one("#personas-library-count", Static)
        assert "2 characters" in str(count.renderable)


async def test_filtered_count_shows_n_of_m():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        pane.update_rows(
            (LibraryRow(item_id="1", kind="character", name="Detective Sam"),),
            total=12,
            noun="characters",
            filtered=True,
        )
        await pilot.pause()
        count = pilot.app.query_one("#personas-library-count", Static)
        assert "1 of 12 characters" in str(count.renderable)


async def test_row_press_posts_library_row_selected():
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_library_row_selected(self, message: LibraryRowSelected) -> None:
            received.append((message.kind, message.item_id))

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        pane.update_rows(
            (LibraryRow(item_id="7", kind="character", name="Detective Sam"),),
            total=1,
            noun="characters",
        )
        await pilot.pause()
        await pilot.click("#personas-library-row-character-7")
        await pilot.pause()
    assert received == [("character", "7")]


async def test_search_input_posts_search_changed_and_new_posts_new():
    searches = []
    news = []

    class CaptureApp(LibraryPaneApp):
        def on_library_search_changed(self, message: LibrarySearchChanged) -> None:
            searches.append(message.query)

        def on_library_new_requested(self, message: LibraryNewRequested) -> None:
            news.append(True)

    app = CaptureApp()
    async with app.run_test() as pilot:
        search = pilot.app.query_one("#personas-library-search", Input)
        search.value = "sam"
        await pilot.pause()
        await pilot.click("#personas-library-new")
        await pilot.pause()
    assert searches[-1] == "sam"
    assert news == [True]
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q Tests/UI/test_personas_library_pane.py --tb=short`
Expected: FAIL with `ModuleNotFoundError` for `personas_library_pane`

- [ ] **Step 3: Implement the pane**

```python
# tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py
"""Mode-scoped library list pane for the Personas workbench."""

from __future__ import annotations

import re
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from .personas_messages import (
    LibraryImportRequested,
    LibraryNewRequested,
    LibraryRowSelected,
    LibrarySearchChanged,
)

_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


def _row_dom_id(kind: str, item_id: str) -> str:
    return f"personas-library-row-{kind}-{_ID_SAFE.sub('-', str(item_id))}"


@dataclass(frozen=True)
class LibraryRow:
    """One selectable row in the workbench library list."""

    item_id: str
    kind: str  # "character" | "persona_profile"
    name: str
    is_unsaved: bool = False


class PersonasLibraryPane(Vertical):
    """Search, create/import toolbar, and selectable item rows."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._row_lookup: dict[str, tuple[str, str]] = {}

    def compose(self) -> ComposeResult:
        yield Static("Library", classes="destination-section personas-column-title")
        yield Input(
            placeholder="Search...",
            id="personas-library-search",
        )
        with Horizontal(id="personas-library-toolbar", classes="ds-toolbar"):
            yield Button("New", id="personas-library-new", tooltip="Create a new item in this mode.")
            yield Button(
                "Import",
                id="personas-library-import",
                tooltip="Import a character card (PNG or JSON).",
            )
        yield VerticalScroll(id="personas-library-rows")
        yield Static("", id="personas-library-count", classes="destination-purpose")

    def set_mode(self, mode: str) -> None:
        """Show Import only where it applies (Characters mode)."""
        self.query_one("#personas-library-import", Button).display = mode == "characters"

    def update_rows(
        self,
        rows: tuple[LibraryRow, ...],
        *,
        total: int,
        noun: str,
        filtered: bool = False,
    ) -> None:
        """Replace the visible rows and count line."""
        container = self.query_one("#personas-library-rows", VerticalScroll)
        container.remove_children()
        self._row_lookup = {}
        if not rows:
            container.mount(
                Static(
                    f"No {noun} yet - use New or Import to add one.",
                    id="personas-library-empty",
                )
            )
        for row in rows:
            dom_id = _row_dom_id(row.kind, row.item_id)
            self._row_lookup[dom_id] = (row.kind, row.item_id)
            classes = "personas-library-row"
            if row.is_unsaved:
                classes += " is-unsaved"
            container.mount(Button(row.name, id=dom_id, classes=classes))
        count = f"{len(rows)} of {total} {noun}" if filtered else f"{total} {noun}"
        self.query_one("#personas-library-count", Static).update(count)

    def mark_active_row(self, kind: str, item_id: str) -> None:
        """Apply .is-active to the selected row only."""
        active_id = _row_dom_id(kind, item_id)
        for button in self.query(".personas-library-row"):
            button.set_class(button.id == active_id, "is-active")

    @on(Button.Pressed, ".personas-library-row")
    def _row_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        entry = self._row_lookup.get(str(event.button.id or ""))
        if entry is not None:
            self.post_message(LibraryRowSelected(kind=entry[0], item_id=entry[1]))

    @on(Button.Pressed, "#personas-library-new")
    def _new_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(LibraryNewRequested())

    @on(Button.Pressed, "#personas-library-import")
    def _import_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(LibraryImportRequested())

    @on(Input.Changed, "#personas-library-search")
    def _search_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.post_message(LibrarySearchChanged(query=event.value))
```

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_library_pane.py --tb=short`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py Tests/UI/test_personas_library_pane.py
git commit -m "feat: add Personas workbench library pane

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 4: Inspector pane widget

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- Test: `Tests/UI/test_personas_inspector_pane.py`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_personas_inspector_pane.py
"""Mounted tests for the Personas inspector pane."""

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    ConversationRowSelected,
)

pytestmark = pytest.mark.asyncio


class InspectorApp(App):
    def compose(self):
        yield PersonasInspectorPane(id="personas-inspector-pane")


async def test_default_state_shows_no_selection_and_disabled_actions():
    app = InspectorApp()
    async with app.run_test() as pilot:
        assert "Selected: none" in str(
            pilot.app.query_one("#personas-selected-name", Static).renderable
        )
        for button_id in (
            "#personas-attach-to-console",
            "#personas-start-chat",
            "#personas-export-json",
            "#personas-export-png",
            "#personas-delete",
        ):
            assert pilot.app.query_one(button_id, Button).disabled is True
        assert "Console: Blocked" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_show_selection_enables_actions_and_shows_authority():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Detective Sam", kind="character", authority="Local")
        await pilot.pause()
        assert "Selected: Detective Sam" in str(
            pilot.app.query_one("#personas-selected-name", Static).renderable
        )
        assert "Authority: Local" in str(
            pilot.app.query_one("#personas-selected-authority", Static).renderable
        )
        assert pilot.app.query_one("#personas-attach-to-console", Button).disabled is False
        assert "Console: Ready" in str(
            pilot.app.query_one("#personas-readiness-console", Static).renderable
        )


async def test_unsaved_disables_attach_export_with_reason():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_selection(name="Tutor", kind="character", authority="Local")
        pane.set_unsaved(True)
        await pilot.pause()
        attach = pilot.app.query_one("#personas-attach-to-console", Button)
        assert attach.disabled is True
        assert "unsaved" in str(attach.tooltip).lower()
        assert pilot.app.query_one("#personas-export-json", Button).disabled is True


async def test_show_validation_errors_renders_messages():
    app = InspectorApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_validation(("name: required", "first_message: required"))
        await pilot.pause()
        summary = str(pilot.app.query_one("#personas-validation-summary", Static).renderable)
        assert "name: required" in summary
        pane.show_validation(())
        await pilot.pause()
        assert "Validation: OK" in str(
            pilot.app.query_one("#personas-validation-summary", Static).renderable
        )


async def test_conversations_panel_rows_post_selection():
    received = []

    class CaptureApp(InspectorApp):
        def on_conversation_row_selected(self, message: ConversationRowSelected) -> None:
            received.append(message.conversation_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasInspectorPane)
        pane.show_conversations((("conv-1", "First case"), ("conv-2", "Cold trail")))
        await pilot.pause()
        assert len(pilot.app.query(".personas-conversation-row")) == 2
        await pilot.click("#personas-conversation-row-0")
        await pilot.pause()
    assert received == ["conv-1"]
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q Tests/UI/test_personas_inspector_pane.py --tb=short`
Expected: FAIL with `ModuleNotFoundError` for `personas_inspector_pane`

- [ ] **Step 3: Implement the pane**

```python
# tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py
"""Selected-item inspector pane for the Personas workbench."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Static

from .personas_messages import ConversationRowSelected

_UNSAVED_TOOLTIP = "Save before using this action; the selection has unsaved edits."


class PersonasInspectorPane(Vertical):
    """Identity, validation, conversations, readiness, and actions."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._has_selection = False
        self._is_unsaved = False
        self._conversation_ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("Inspector", classes="destination-section personas-column-title")
        yield Static("Selected: none", id="personas-selected-name")
        yield Static("Type: -", id="personas-selected-kind")
        yield Static("Authority: Local", id="personas-selected-authority")
        yield Static("Validation: OK", id="personas-validation-summary")
        yield Static("Conversations", classes="destination-section")
        yield VerticalScroll(id="personas-conversations-list")
        yield Static("Readiness", classes="destination-section")
        yield Static("Console: Blocked - select an item", id="personas-readiness-console")
        with Vertical(id="personas-inspector-actions"):
            yield Button("Attach to Console", id="personas-attach-to-console", disabled=True)
            yield Button("Start Chat", id="personas-start-chat", disabled=True)
            yield Button("Export JSON", id="personas-export-json", disabled=True)
            yield Button("Export PNG", id="personas-export-png", disabled=True)
            yield Button(
                "Delete",
                id="personas-delete",
                disabled=True,
                classes="personas-destructive",
            )

    def show_selection(self, *, name: str, kind: str, authority: str) -> None:
        self._has_selection = True
        self.query_one("#personas-selected-name", Static).update(f"Selected: {name}")
        self.query_one("#personas-selected-kind", Static).update(f"Type: {kind}")
        self.query_one("#personas-selected-authority", Static).update(f"Authority: {authority}")
        self._apply_action_state(kind)

    def clear_selection(self) -> None:
        self._has_selection = False
        self._is_unsaved = False
        self.query_one("#personas-selected-name", Static).update("Selected: none")
        self.query_one("#personas-selected-kind", Static).update("Type: -")
        self.show_conversations(())
        self.show_validation(())
        self._apply_action_state(kind=None)

    def set_unsaved(self, is_unsaved: bool) -> None:
        self._is_unsaved = is_unsaved
        kind = str(self.query_one("#personas-selected-kind", Static).renderable).removeprefix("Type: ")
        self._apply_action_state(kind if self._has_selection else None)

    def show_validation(self, errors: tuple[str, ...]) -> None:
        summary = self.query_one("#personas-validation-summary", Static)
        if errors:
            summary.update("Validation errors:\n" + "\n".join(errors))
        else:
            summary.update("Validation: OK")

    def show_conversations(self, rows: tuple[tuple[str, str], ...]) -> None:
        """Render (conversation_id, title) rows; hide section when empty."""
        container = self.query_one("#personas-conversations-list", VerticalScroll)
        container.remove_children()
        self._conversation_ids = [conversation_id for conversation_id, _title in rows]
        for index, (_conversation_id, title) in enumerate(rows):
            container.mount(
                Button(
                    title,
                    id=f"personas-conversation-row-{index}",
                    classes="personas-conversation-row",
                )
            )

    def _apply_action_state(self, kind: str | None) -> None:
        selected = self._has_selection
        unsaved = self._is_unsaved
        readiness = self.query_one("#personas-readiness-console", Static)
        if not selected:
            readiness.update("Console: Blocked - select an item")
        elif unsaved:
            readiness.update("Console: Blocked - unsaved edits")
        else:
            readiness.update("Console: Ready")
        attach_export_enabled = selected and not unsaved
        for button_id in ("#personas-attach-to-console", "#personas-start-chat"):
            button = self.query_one(button_id, Button)
            button.disabled = not attach_export_enabled
            button.tooltip = _UNSAVED_TOOLTIP if (selected and unsaved) else None
        png_allowed = attach_export_enabled and kind == "character"
        json_button = self.query_one("#personas-export-json", Button)
        json_button.disabled = not attach_export_enabled
        json_button.tooltip = _UNSAVED_TOOLTIP if (selected and unsaved) else None
        self.query_one("#personas-export-png", Button).disabled = not png_allowed
        self.query_one("#personas-delete", Button).disabled = not selected

    @on(Button.Pressed, ".personas-conversation-row")
    def _conversation_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id or "")
        index = int(button_id.rsplit("-", 1)[-1])
        if 0 <= index < len(self._conversation_ids):
            self.post_message(ConversationRowSelected(self._conversation_ids[index]))
```

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_inspector_pane.py --tb=short`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py Tests/UI/test_personas_inspector_pane.py
git commit -m "feat: add Personas workbench inspector pane

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 5: Rebuild PersonasScreen as the workbench shell (Characters mode)

The big task: replace the thin snapshot shell with the workbench. Characters mode lists,
views, edits, creates, and deletes characters through the reused CCP behavior layer.

**Files:**
- Rewrite: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_workbench.py`
- Reference (read, do not modify): `tldw_chatbook/UI/Screens/notes_screen.py`,
  `tldw_chatbook/UI/Screens/ccp_screen.py`, `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py`

**Key reuse facts (verified):**
- `CCPCharacterHandler(self)` stores the screen as `self.window`; it calls
  `self.window.refresh_character_library_list(characters)` when present, runs DB loads
  via `self.window.run_worker`, and renders the loaded card into `#ccp-card-*` statics
  inside `CCPCharacterCardWidget`. Its `#conv-char-character-select` query is already
  guarded with try/except.
- Module-level functions in `ccp_character_handler.py`: `fetch_all_characters()`,
  `fetch_character_by_id(character_id)`, `create_character(data)`,
  `update_character(character_id, data)`, `import_character_card(file_path)`.
- `CCPCharacterCardWidget` posts `EditCharacterRequested(character_id)`; it and
  `CCPCharacterEditorWidget` set their own internal view IDs
  (`#ccp-character-card-view`, `#ccp-character-editor-view`).
- `CCPCharacterEditorWidget` exposes `load_character(character_data)` and
  `new_character()`, and posts `CharacterSaveRequested(character_data)` /
  `CharacterEditorCancelled`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_personas_workbench.py
"""Mounted tests for the destination-native Personas workbench."""

import pytest
from textual.app import App
from textual.widgets import Button, Static

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

pytestmark = pytest.mark.asyncio

CHARACTERS = [
    {"id": 1, "name": "Detective Sam", "description": "Noir detective", "version": 1},
    {"id": 2, "name": "Lab Assistant", "description": "Helpful scientist", "version": 1},
]


@pytest.fixture
def stub_characters(monkeypatch):
    monkeypatch.setattr(
        character_handler_module, "fetch_all_characters", lambda: list(CHARACTERS)
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            dict(c) for c in CHARACTERS if str(c["id"]) == str(character_id)
        ),
    )


class PersonasTestApp(App):
    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = mock_app_instance.character_persona_scope_service

    def __getattr__(self, name):
        return getattr(self._mock, name)

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


async def _mounted(pilot):
    await pilot.pause()
    return pilot.app.screen


class TestWorkbenchShell:
    async def test_route_renders_destination_workbench(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            title = screen.query_one("#personas-title", Static)
            assert "Personas" in str(title.renderable)
            assert "ds-destination-header" in title.classes
            assert screen.query_one("#personas-mode-strip")
            assert screen.query_one("#personas-library-pane")
            assert screen.query_one("#personas-work-area")
            assert screen.query_one("#personas-inspector-pane")

    async def test_characters_mode_lists_library_rows(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [str(r.label) for r in rows] == ["Detective Sam", "Lab Assistant"]

    async def test_placeholder_modes_show_placeholder_panel(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-mode-prompts")
            await pilot.pause()
            assert screen.state.active_mode == "prompts"
            placeholder = screen.query_one("#personas-mode-placeholder", Static)
            assert "not available yet" in str(placeholder.renderable)
            assert "is-active" in screen.query_one("#personas-mode-prompts", Button).classes


class TestCharacterSelectionAndEdit:
    async def test_row_selection_shows_card_and_inspector(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            assert screen.state.selected_id == "1"
            assert screen.state.edit_mode == "view"
            assert "Selected: Detective Sam" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )

    async def test_new_button_opens_editor_in_create_mode(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            assert screen.state.edit_mode == "create"
            editor = screen.query_one("#ccp-character-editor-view")
            assert editor.display is True

    async def test_save_with_missing_name_blocks_and_shows_validation(self, mock_app_instance, stub_characters, monkeypatch):
        created = []
        monkeypatch.setattr(
            character_handler_module, "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_editor_widget import (
                CharacterSaveRequested,
            )
            screen.post_message(CharacterSaveRequested({"name": "", "first_message": "Hi"}))
            await pilot.pause()
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "name: required" in str(summary.renderable)
        assert created == []

    async def test_save_calls_create_and_refreshes(self, mock_app_instance, stub_characters, monkeypatch):
        created = []
        monkeypatch.setattr(
            character_handler_module,
            "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_editor_widget import (
                CharacterSaveRequested,
            )
            screen.post_message(CharacterSaveRequested({"name": "New Hero", "first_message": "Hi"}))
            await pilot.pause()
            await pilot.pause()
        assert created and created[0]["name"] == "New Hero"
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`
Expected: FAIL — `PersonasScreen` has no `state`/workbench IDs (current thin shell).

- [ ] **Step 3: Rewrite the screen**

Replace the entire contents of `tldw_chatbook/UI/Screens/personas_screen.py`:

```python
# tldw_chatbook/UI/Screens/personas_screen.py
"""Destination-native Personas workbench: characters, personas, prompts, lore."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Widgets.CCP_Widgets import CCPCharacterCardWidget, CCPCharacterEditorWidget
from ...Widgets.CCP_Widgets.ccp_character_card_widget import EditCharacterRequested
from ...Widgets.CCP_Widgets.ccp_character_editor_widget import (
    CharacterEditorCancelled,
    CharacterSaveRequested,
)
from ...Widgets.confirmation_dialog import UnsavedChangesDialog
from ...Widgets.destination_workbench import DestinationModeStrip
from ...Widgets.Persona_Widgets.personas_inspector_pane import PersonasInspectorPane
from ...Widgets.Persona_Widgets.personas_library_pane import (
    LibraryRow,
    PersonasLibraryPane,
)
from ...Widgets.Persona_Widgets.personas_messages import (
    LibraryNewRequested,
    LibraryRowSelected,
)
from ...Widgets.Persona_Widgets.personas_state import (
    PERSONAS_MODES,
    PLACEHOLDER_MODES,
    PersonasWorkbenchState,
)
from ..CCP_Modules.ccp_character_handler import CCPCharacterHandler
from ..CCP_Modules.ccp_persona_handler import CCPPersonaHandler
from ..Navigation.base_app_screen import BaseAppScreen

logger = logger.bind(module="PersonasScreen")

MODE_LABELS: dict[str, str] = {
    "characters": "Characters",
    "personas": "Personas",
    "prompts": "Prompts",
    "dictionaries": "Dictionaries",
    "lore": "Lore",
}
MODE_NOUNS: dict[str, str] = {"characters": "characters", "personas": "persona profiles"}


class PersonasScreen(BaseAppScreen):
    """Characters, personas, prompts, dictionaries, and behavior profiles."""

    state: reactive[PersonasWorkbenchState] = reactive(PersonasWorkbenchState())

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "personas", **kwargs)
        self.character_handler = CCPCharacterHandler(self)
        self.persona_handler = CCPPersonaHandler(self)
        self._characters: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------ compose

    def compose_content(self) -> ComposeResult:
        with Vertical(id="personas-shell"):
            yield Static(
                "Personas | Behavior profiles for chat and agents | Ready | Local",
                id="personas-title",
                classes="ds-destination-header",
            )
            yield Static(
                "Create, edit, and manage characters and persona profiles; "
                "attach them to Console.",
                id="personas-purpose",
                classes="destination-purpose",
            )
            yield Static(self._status_row_text(), id="personas-status-row",
                         classes="destination-status-row")
            with DestinationModeStrip(id="personas-mode-strip", classes="destination-mode-strip"):
                yield Static("Modes:", id="personas-mode-label", classes="destination-section")
                for mode in PERSONAS_MODES:
                    classes = "personas-mode-chip"
                    if mode == self.state.active_mode:
                        classes += " is-active"
                    yield Button(MODE_LABELS[mode], id=f"personas-mode-{mode}", classes=classes)
            with Horizontal(id="personas-workbench", classes="ds-panel destination-workbench"):
                yield PersonasLibraryPane(
                    id="personas-library-pane",
                    classes="destination-workbench-pane",
                )
                with Vertical(id="personas-work-area", classes="destination-workbench-pane"):
                    with Container(id="personas-detail-stack"):
                        yield CCPCharacterCardWidget(parent_screen=self)
                        yield CCPCharacterEditorWidget(parent_screen=self)
                        yield Static(
                            "This mode is not available yet. Characters and Personas "
                            "are the supported modes.",
                            id="personas-mode-placeholder",
                        )
                yield PersonasInspectorPane(
                    id="personas-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                )

    async def on_mount(self) -> None:
        super().on_mount()
        self.query_one(PersonasLibraryPane).set_mode(self.state.active_mode)
        self._show_center(None)
        await self.character_handler.refresh_character_list()

    # ----------------------------------------------------------- handler hooks

    async def refresh_character_library_list(self, characters: list[Dict[str, Any]]) -> None:
        """Hook called by CCPCharacterHandler after fetching characters."""
        self._characters = list(characters or [])
        if self.state.active_mode == "characters":
            self._render_library_rows()

    # ------------------------------------------------------------- mode switch

    @on(Button.Pressed, ".personas-mode-chip")
    async def _mode_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        mode = str(event.button.id or "").removeprefix("personas-mode-")
        if mode == self.state.active_mode:
            return
        if not await self._confirm_discard_unsaved():
            return
        self.state = self.state.with_mode(mode)
        for chip in self.query(".personas-mode-chip"):
            chip.set_class(chip.id == f"personas-mode-{mode}", "is-active")
        self.query_one("#personas-status-row", Static).update(self._status_row_text())
        library = self.query_one(PersonasLibraryPane)
        library.set_mode(mode)
        self.query_one(PersonasInspectorPane).clear_selection()
        if mode in PLACEHOLDER_MODES:
            library.update_rows((), total=0, noun=MODE_LABELS[mode].lower())
            self._show_center("#personas-mode-placeholder")
        elif mode == "characters":
            self._render_library_rows()
            self._show_center(None)
        elif mode == "personas":
            library.update_rows((), total=0, noun=MODE_NOUNS["personas"])
            self._show_center(None)

    # ------------------------------------------------------ selection and edit

    @on(LibraryRowSelected)
    async def _row_selected(self, message: LibraryRowSelected) -> None:
        if not await self._confirm_discard_unsaved():
            return
        if message.kind == "character":
            from dataclasses import replace
            self.state = replace(
                self.state, selected_kind="character",
                selected_id=message.item_id, edit_mode="view", is_unsaved=False,
            )
            self.query_one(PersonasLibraryPane).mark_active_row("character", message.item_id)
            await self.character_handler.load_character(message.item_id)
            self._show_center("#ccp-character-card-view")
            record = self._character_record(message.item_id)
            self.query_one(PersonasInspectorPane).show_selection(
                name=str(record.get("name", "Unnamed")) if record else "Unnamed",
                kind="character",
                authority="Local",
            )

    @on(LibraryNewRequested)
    async def _new_requested(self, message: LibraryNewRequested) -> None:
        if not await self._confirm_discard_unsaved():
            return
        if self.state.active_mode == "characters":
            from dataclasses import replace
            self.state = replace(self.state, edit_mode="create", selected_kind="character",
                                 selected_id=None, is_unsaved=True)
            editor = self.query_one(CCPCharacterEditorWidget)
            editor.new_character()
            self._show_center("#ccp-character-editor-view")
            self.query_one(PersonasInspectorPane).set_unsaved(True)

    @on(EditCharacterRequested)
    def _edit_requested(self, message: EditCharacterRequested) -> None:
        from dataclasses import replace
        record = self._character_record(str(message.character_id))
        if record is None:
            return
        self.state = replace(self.state, edit_mode="edit", is_unsaved=True)
        editor = self.query_one(CCPCharacterEditorWidget)
        editor.load_character(record)
        self._show_center("#ccp-character-editor-view")
        self.query_one(PersonasInspectorPane).set_unsaved(True)

    @on(CharacterSaveRequested)
    def _save_requested(self, message: CharacterSaveRequested) -> None:
        data = dict(message.character_data)
        errors = self._validate_character(data)
        self.query_one(PersonasInspectorPane).show_validation(errors)
        if errors:
            return
        self._save_character(data)

    def _validate_character(self, data: Dict[str, Any]) -> tuple[str, ...]:
        """Field-level validation; failures block Save and render in the inspector."""
        errors: list[str] = []
        if not str(data.get("name", "")).strip():
            errors.append("name: required")
        book = data.get("character_book")
        if book:
            from ...Character_Chat.Character_Chat_Lib import validate_character_book
            ok, book_errors = validate_character_book(book)
            if not ok:
                errors.extend(str(e) for e in book_errors)
        return tuple(errors)

    @work(thread=True, exclusive=True)
    def _save_character(self, data: Dict[str, Any]) -> None:
        from ..CCP_Modules import ccp_character_handler as handler_module
        try:
            if self.state.edit_mode == "create" or self.state.selected_id is None:
                new_id = handler_module.create_character(data)
                saved_id = str(new_id)
            else:
                handler_module.update_character(self.state.selected_id, data)
                saved_id = self.state.selected_id
        except Exception as exc:
            logger.error(f"Character save failed: {exc}", exc_info=True)
            self.app.call_from_thread(self._notify, f"Save failed: {exc}", "error")
            return
        self.app.call_from_thread(self._after_character_save, saved_id)

    async def _after_character_save(self, saved_id: str) -> None:
        from dataclasses import replace
        self.state = replace(self.state, edit_mode="view", is_unsaved=False,
                             selected_id=saved_id, selected_kind="character")
        self.query_one(PersonasInspectorPane).set_unsaved(False)
        self.query_one(PersonasInspectorPane).show_validation(())
        await self.character_handler.refresh_character_list()
        await self.character_handler.load_character(saved_id)
        self._show_center("#ccp-character-card-view")
        self._notify("Character saved.", "information")

    @on(CharacterEditorCancelled)
    async def _edit_cancelled(self, message: CharacterEditorCancelled) -> None:
        if not await self._confirm_discard_unsaved():
            return
        from dataclasses import replace
        self.state = replace(self.state, edit_mode="view", is_unsaved=False)
        self.query_one(PersonasInspectorPane).set_unsaved(False)
        if self.state.selected_id is not None:
            self._show_center("#ccp-character-card-view")
        else:
            self._show_center(None)

    # ------------------------------------------------------------------ helpers

    def _status_row_text(self) -> str:
        label = MODE_LABELS[self.state.active_mode]
        return f"Mode: {label} | Source: Local | Attachments: Console"

    def _render_library_rows(self) -> None:
        rows = tuple(
            LibraryRow(
                item_id=str(record.get("id", "")),
                kind="character",
                name=str(record.get("name", "Unnamed")),
            )
            for record in self._characters
        )
        self.query_one(PersonasLibraryPane).update_rows(
            rows, total=len(rows), noun=MODE_NOUNS["characters"]
        )

    def _character_record(self, item_id: str) -> Dict[str, Any] | None:
        for record in self._characters:
            if str(record.get("id")) == str(item_id):
                return record
        return None

    def _show_center(self, visible_id: str | None) -> None:
        """Show one child of the detail stack; hide the rest."""
        for child_id in (
            "#ccp-character-card-view",
            "#ccp-character-editor-view",
            "#personas-mode-placeholder",
        ):
            try:
                widget = self.query_one(child_id)
            except Exception:
                continue
            widget.display = child_id == visible_id

    async def _confirm_discard_unsaved(self) -> bool:
        """True when navigation may proceed (no unsaved edits, or user discards)."""
        if not self.state.is_unsaved:
            return True
        return bool(
            await self.app.push_screen_wait(
                UnsavedChangesDialog(message="Discard unsaved changes?")
            )
        )

    def _notify(self, message: str, severity: str = "warning") -> None:
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity=severity)
```

Note: `UnsavedChangesDialog.__init__` — check its actual signature in
`tldw_chatbook/Widgets/confirmation_dialog.py:139` before wiring; pass parameters it
accepts (it subclasses `ConfirmationDialog`, which takes `message`). Adjust the call if
the constructor differs, keeping the await-for-bool contract.

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py Tests/UI/test_personas_workbench_state.py --tb=short`
Expected: all pass

- [ ] **Step 5: Run neighboring suites to catch fallout**

Run: `python -m pytest -q Tests/UI/test_ccp_screen.py Tests/UI/test_personas_library_pane.py Tests/UI/test_personas_inspector_pane.py --tb=short`
Expected: pass (legacy ccp route untouched in this phase). If anything imported the old
thin-shell symbols (`PERSONAS_EMPTY_COPY` etc.), update those imports.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat: rebuild Personas route as destination-native workbench (Characters mode)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 6: Footer shortcut context

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Write the failing test** (append to `TestWorkbenchShell`)

```python
    async def test_footer_shortcut_context_set_and_cleared(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            context = screen._shortcut_context()
            rendered = context.render()
            assert "new" in rendered.lower()
            assert "search" in rendered.lower()
            assert context.source == "personas"
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q "Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_footer_shortcut_context_set_and_cleared" --tb=short`
Expected: FAIL — no `_shortcut_context`

- [ ] **Step 3: Implement**

Add to `personas_screen.py` (imports at top, methods on the class):

```python
from ..Navigation.shortcut_context import ShortcutAction, ShortcutContext
```

```python
    def _shortcut_context(self) -> ShortcutContext:
        return ShortcutContext(
            source="personas",
            actions=(
                ShortcutAction("ctrl+n", "new"),
                ShortcutAction("ctrl+f", "search"),
                ShortcutAction("ctrl+s", "save"),
                ShortcutAction("ctrl+enter", "attach"),
            ),
        )

    def _register_footer_shortcuts(self) -> None:
        footer = getattr(self.app_instance, "footer_status", None)
        set_context = getattr(footer, "set_shortcut_context", None)
        if callable(set_context):
            set_context(self._shortcut_context())

    def _clear_footer_shortcuts(self) -> None:
        footer = getattr(self.app_instance, "footer_status", None)
        clear_context = getattr(footer, "clear_shortcut_context", None)
        if callable(clear_context):
            clear_context()
```

Call `self._register_footer_shortcuts()` at the end of `on_mount`, and add:

```python
    def on_unmount(self) -> None:
        self._clear_footer_shortcuts()
        super().on_unmount()
```

Before finalizing key choices, run `grep -rn "ctrl+n\|ctrl+f\|ctrl+s" tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/app.py`
and change any key that collides with a global binding (spec: keys are illustrative).

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat: register Personas workbench footer shortcut context

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Phase 1 exit criteria:** Characters mode works end-to-end (list, view, create, edit,
save, cancel, unsaved guard); legacy `ccp` route still works; `python -m pytest -q Tests/UI --tb=short` passes.

---

## Phase 2 — Personas mode

### Task 7: Persona profile card and editor widgets

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_card_widget.py`
- Create: `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py`
- Test: `Tests/UI/test_persona_profile_widgets.py`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_persona_profile_widgets.py
"""Mounted tests for persona profile card and editor widgets."""

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_card_widget import (
    PersonaProfileCardWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    EditPersonaRequested,
    PersonaProfileSaveRequested,
)

pytestmark = pytest.mark.asyncio

PROFILE = {
    "id": "p-1",
    "name": "Archivist",
    "description": "Preserve, organize, retrieve",
    "system_prompt": "You are a meticulous archivist.",
}


class WidgetApp(App):
    def compose(self):
        yield PersonaProfileCardWidget()
        yield PersonaProfileEditorWidget()


async def test_card_shows_profile_and_edit_posts_message():
    received = []

    class CaptureApp(WidgetApp):
        def on_edit_persona_requested(self, message: EditPersonaRequested) -> None:
            received.append(message.persona_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        card = pilot.app.query_one(PersonaProfileCardWidget)
        card.show_persona(PROFILE)
        await pilot.pause()
        assert "Archivist" in str(
            pilot.app.query_one("#personas-card-name", Static).renderable
        )
        assert "meticulous archivist" in str(
            pilot.app.query_one("#personas-card-system-prompt", Static).renderable
        )
        await pilot.click("#personas-card-edit")
        await pilot.pause()
    assert received == ["p-1"]


async def test_editor_load_collect_roundtrip():
    app = WidgetApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.load_persona(PROFILE)
        await pilot.pause()
        assert pilot.app.query_one("#personas-editor-name", Input).value == "Archivist"
        data = editor.collect()
        assert data["name"] == "Archivist"
        assert data["system_prompt"] == "You are a meticulous archivist."


async def test_editor_save_posts_collected_data():
    received = []

    class CaptureApp(WidgetApp):
        def on_persona_profile_save_requested(self, message: PersonaProfileSaveRequested) -> None:
            received.append(message.data)

    app = CaptureApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.new_persona()
        pilot.app.query_one("#personas-editor-name", Input).value = "Mentor"
        await pilot.pause()
        await pilot.click("#personas-editor-save")
        await pilot.pause()
    assert received and received[0]["name"] == "Mentor"


async def test_editor_save_requires_name():
    app = WidgetApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.new_persona()
        await pilot.pause()
        errors = editor.validate()
        assert errors == ("name: required",)
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q Tests/UI/test_persona_profile_widgets.py --tb=short`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the widgets**

```python
# tldw_chatbook/Widgets/Persona_Widgets/persona_profile_card_widget.py
"""Read-only persona profile view for the Personas workbench."""

from __future__ import annotations

from typing import Any, Dict

from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Button, Static

from .personas_messages import EditPersonaRequested


class PersonaProfileCardWidget(Container):
    """Read-only persona profile card with an Edit action."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("id", "ccp-persona-card-view")
        super().__init__(**kwargs)
        self._persona_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Persona Profile", classes="destination-section")
        yield Static("", id="personas-card-name")
        yield Static("", id="personas-card-description")
        yield Static("System prompt", classes="destination-section")
        yield Static("", id="personas-card-system-prompt")
        yield Button("Edit", id="personas-card-edit")

    def show_persona(self, data: Dict[str, Any]) -> None:
        self._persona_id = str(data.get("id", "")) or None
        self.query_one("#personas-card-name", Static).update(str(data.get("name", "Unnamed")))
        self.query_one("#personas-card-description", Static).update(
            str(data.get("description", ""))
        )
        self.query_one("#personas-card-system-prompt", Static).update(
            str(data.get("system_prompt", ""))
        )

    @on(Button.Pressed, "#personas-card-edit")
    def _edit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._persona_id is not None:
            self.post_message(EditPersonaRequested(self._persona_id))
```

```python
# tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py
"""Persona profile create/edit form for the Personas workbench."""

from __future__ import annotations

from typing import Any, Dict

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Label, Static, TextArea

from .personas_messages import PersonaProfileEditCancelled, PersonaProfileSaveRequested


class PersonaProfileEditorWidget(Container):
    """ds-field-row form: name, description, system prompt."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("id", "ccp-persona-editor-view")
        super().__init__(**kwargs)
        self._persona_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Persona Editor", classes="destination-section")
        with Vertical(classes="ds-field-row"):
            yield Label("Name")
            yield Input(id="personas-editor-name", placeholder="Persona name")
        with Vertical(classes="ds-field-row"):
            yield Label("Description")
            yield TextArea(id="personas-editor-description")
        with Vertical(classes="ds-field-row"):
            yield Label("System prompt")
            yield TextArea(id="personas-editor-system-prompt")
        yield Static("", id="personas-editor-validation")
        with Horizontal(classes="ds-toolbar"):
            yield Button("Save", id="personas-editor-save")
            yield Button("Cancel", id="personas-editor-cancel")

    # Handler compatibility hook: CCPPersonaHandler calls load_persona when present.
    def load_persona(self, data: Dict[str, Any]) -> None:
        self._persona_id = str(data.get("id", "")) or None
        self.query_one("#personas-editor-name", Input).value = str(data.get("name", ""))
        self.query_one("#personas-editor-description", TextArea).text = str(
            data.get("description", "")
        )
        self.query_one("#personas-editor-system-prompt", TextArea).text = str(
            data.get("system_prompt", "")
        )

    def new_persona(self) -> None:
        self.load_persona({})

    def collect(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.query_one("#personas-editor-name", Input).value.strip(),
            "description": self.query_one("#personas-editor-description", TextArea).text,
            "system_prompt": self.query_one("#personas-editor-system-prompt", TextArea).text,
        }
        if self._persona_id is not None:
            data["id"] = self._persona_id
        return data

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.query_one("#personas-editor-name", Input).value.strip():
            errors.append("name: required")
        return tuple(errors)

    @on(Button.Pressed, "#personas-editor-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        errors = self.validate()
        validation = self.query_one("#personas-editor-validation", Static)
        if errors:
            validation.update("Validation errors:\n" + "\n".join(errors))
            return
        validation.update("")
        self.post_message(PersonaProfileSaveRequested(self.collect()))

    @on(Button.Pressed, "#personas-editor-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaProfileEditCancelled())
```

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_persona_profile_widgets.py --tb=short`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/persona_profile_card_widget.py \
        tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py \
        Tests/UI/test_persona_profile_widgets.py
git commit -m "feat: add persona profile card and editor widgets

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 8: Wire Personas mode into the screen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Write the failing tests** (new class in `test_personas_workbench.py`)

```python
PROFILES = [
    {"id": "p-1", "name": "Archivist", "description": "Preserve and retrieve",
     "system_prompt": "You are a meticulous archivist."},
]


class TestPersonasMode:
    async def test_personas_mode_lists_profiles(self, mock_app_instance, stub_characters):
        mock_app_instance.character_persona_scope_service.list_persona_profiles.return_value = {
            "items": PROFILES, "total": 1,
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [str(r.label) for r in rows] == ["Archivist"]

    async def test_profile_selection_shows_card(self, mock_app_instance, stub_characters):
        service = mock_app_instance.character_persona_scope_service
        service.list_persona_profiles.return_value = {"items": PROFILES, "total": 1}
        service.get_persona_profile.return_value = PROFILES[0]
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#personas-library-row-persona_profile-p-1")
            await pilot.pause()
            await pilot.pause()
            assert screen.state.selected_kind == "persona_profile"
            card = screen.query_one("#ccp-persona-card-view")
            assert card.display is True

    async def test_profile_save_calls_scope_service(self, mock_app_instance, stub_characters):
        service = mock_app_instance.character_persona_scope_service
        service.list_persona_profiles.return_value = {"items": [], "total": 0}
        service.create_persona_profile.return_value = {"id": "p-9", "name": "Mentor"}
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
                PersonaProfileSaveRequested,
            )
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await pilot.pause()
        service.create_persona_profile.assert_called_once()
```

If `mock_app_instance.character_persona_scope_service` is not already an
`AsyncMock`-style fixture, set the needed methods with
`unittest.mock.AsyncMock(return_value=...)` inside the tests instead of plain
`return_value` assignment — match how `Tests/UI/test_ccp_screen.py` stubs the same
service.

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q "Tests/UI/test_personas_workbench.py::TestPersonasMode" --tb=short`
Expected: FAIL — personas mode renders nothing

- [ ] **Step 3: Implement personas mode in the screen**

In `personas_screen.py`:

1. Add imports:

```python
from ...Widgets.Persona_Widgets.persona_profile_card_widget import PersonaProfileCardWidget
from ...Widgets.Persona_Widgets.persona_profile_editor_widget import PersonaProfileEditorWidget
from ...Widgets.Persona_Widgets.personas_messages import (
    EditPersonaRequested,
    PersonaProfileEditCancelled,
    PersonaProfileSaveRequested,
)
```

2. Add the two widgets to `#personas-detail-stack` in `compose_content`, after the
character widgets:

```python
                        yield PersonaProfileCardWidget()
                        yield PersonaProfileEditorWidget()
```

3. Extend `_show_center`'s id tuple with `"#ccp-persona-card-view"` and
`"#ccp-persona-editor-view"`.

4. In `_mode_pressed`, replace the `elif mode == "personas":` branch body with:

```python
        elif mode == "personas":
            self._refresh_persona_rows()
            self._show_center(None)
```

5. Add persona handling methods to the class:

```python
    _profiles: list[Dict[str, Any]]

    @work(exclusive=True)
    async def _refresh_persona_rows(self) -> None:
        profiles = await self.persona_handler.refresh_persona_list()
        self._profiles = list(profiles or [])
        if self.state.active_mode != "personas":
            return
        rows = tuple(
            LibraryRow(
                item_id=str(record.get("id", "")),
                kind="persona_profile",
                name=str(record.get("name", "Unnamed")),
            )
            for record in self._profiles
        )
        self.query_one(PersonasLibraryPane).update_rows(
            rows, total=len(rows), noun=MODE_NOUNS["personas"]
        )
```

(Initialize `self._profiles = []` in `__init__`.)

6. Extend `_row_selected` with a persona branch:

```python
        elif message.kind == "persona_profile":
            from dataclasses import replace
            self.state = replace(
                self.state, selected_kind="persona_profile",
                selected_id=message.item_id, edit_mode="view", is_unsaved=False,
            )
            self.query_one(PersonasLibraryPane).mark_active_row(
                "persona_profile", message.item_id
            )
            record = await self._fetch_profile(message.item_id)
            if record is not None:
                self.query_one(PersonaProfileCardWidget).show_persona(record)
                self._show_center("#ccp-persona-card-view")
                self.query_one(PersonasInspectorPane).show_selection(
                    name=str(record.get("name", "Unnamed")),
                    kind="persona_profile",
                    authority="Local",
                )

    async def _fetch_profile(self, persona_id: str) -> Dict[str, Any] | None:
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        get_profile = getattr(service, "get_persona_profile", None)
        if not callable(get_profile):
            return self._profile_record(persona_id)
        try:
            result = await get_profile(persona_id, mode="local")
        except Exception as exc:
            logger.warning(f"Persona profile fetch failed: {exc}")
            return self._profile_record(persona_id)
        return result if isinstance(result, Mapping) else self._profile_record(persona_id)

    def _profile_record(self, persona_id: str) -> Dict[str, Any] | None:
        for record in self._profiles:
            if str(record.get("id")) == str(persona_id):
                return record
        return None
```

7. Extend `_new_requested` with a personas branch mirroring characters
(`new_persona()` on `PersonaProfileEditorWidget`, `_show_center("#ccp-persona-editor-view")`,
unsaved flags), and add save/cancel/edit handlers:

```python
    @on(EditPersonaRequested)
    async def _edit_persona(self, message: EditPersonaRequested) -> None:
        from dataclasses import replace
        record = await self._fetch_profile(message.persona_id)
        if record is None:
            return
        self.state = replace(self.state, edit_mode="edit", is_unsaved=True)
        self.query_one(PersonaProfileEditorWidget).load_persona(record)
        self._show_center("#ccp-persona-editor-view")
        self.query_one(PersonasInspectorPane).set_unsaved(True)

    @on(PersonaProfileSaveRequested)
    async def _save_persona(self, message: PersonaProfileSaveRequested) -> None:
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        try:
            if self.state.edit_mode == "create" or "id" not in message.data:
                result = await service.create_persona_profile(message.data, mode="local")
            else:
                result = await service.update_persona_profile(
                    message.data["id"], message.data, mode="local"
                )
        except Exception as exc:
            logger.error(f"Persona save failed: {exc}", exc_info=True)
            self._notify(f"Save failed: {exc}", "error")
            return
        from dataclasses import replace
        saved = result if isinstance(result, Mapping) else message.data
        self.state = replace(
            self.state, edit_mode="view", is_unsaved=False,
            selected_kind="persona_profile",
            selected_id=str(saved.get("id", message.data.get("id", ""))) or None,
        )
        self.query_one(PersonasInspectorPane).set_unsaved(False)
        self._refresh_persona_rows()
        self.query_one(PersonaProfileCardWidget).show_persona(dict(saved))
        self._show_center("#ccp-persona-card-view")
        self._notify("Persona saved.", "information")

    @on(PersonaProfileEditCancelled)
    async def _cancel_persona_edit(self, message: PersonaProfileEditCancelled) -> None:
        if not await self._confirm_discard_unsaved():
            return
        from dataclasses import replace
        self.state = replace(self.state, edit_mode="view", is_unsaved=False)
        self.query_one(PersonasInspectorPane).set_unsaved(False)
        self._show_center(
            "#ccp-persona-card-view" if self.state.selected_id else None
        )
```

Make `_row_selected` `async` already (it is) and keep `dataclasses.replace` imports
local for consistency with the existing methods.

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat: wire Personas mode CRUD through the scope service

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Phase 2 exit criteria:** both modes create/view/edit/save; `python -m pytest -q Tests/UI --tb=short` passes.

---

## Phase 3 — Search, import/export, conversations, preview, Console actions

### Task 9: Library search/filter

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py` (add module-level `search_characters`)
- Modify: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Write the failing tests** (new class)

```python
class TestSearch:
    async def test_search_filters_loaded_characters_locally(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            from textual.widgets import Input
            search = screen.query_one("#personas-library-search", Input)
            search.value = "sam"
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [str(r.label) for r in rows] == ["Detective Sam"]
            count = screen.query_one("#personas-library-count", Static)
            assert "1 of 2 characters" in str(count.renderable)

    async def test_clearing_search_restores_all_rows(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            from textual.widgets import Input
            search = screen.query_one("#personas-library-search", Input)
            search.value = "sam"
            await pilot.pause()
            search.value = ""
            await pilot.pause()
            assert len(screen.query(".personas-library-row")) == 2
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q "Tests/UI/test_personas_workbench.py::TestSearch" --tb=short`
Expected: FAIL — search does nothing

- [ ] **Step 3: Implement**

Add to `ccp_character_handler.py` next to `fetch_all_characters` (module level):

```python
def search_characters_fts(search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
    """FTS search over character cards for large libraries."""
    db = _default_character_db()
    if db is None:
        return []
    return [
        _normalize_character_payload(row)
        for row in db.search_character_cards(search_term, limit=limit)
    ]
```

In `personas_screen.py`, import `LibrarySearchChanged` from `personas_messages` and add:

```python
    # Page size above which the loaded list may be truncated and FTS is used.
    LIBRARY_FTS_THRESHOLD = 1000

    @on(LibrarySearchChanged)
    def _search_changed(self, message: LibrarySearchChanged) -> None:
        from dataclasses import replace
        self.state = replace(self.state, search_query=message.query.strip())
        if self.state.active_mode == "characters":
            self._render_library_rows()
        elif self.state.active_mode == "personas":
            self._render_profile_rows_filtered()
```

Change `_render_library_rows` to apply the filter:

```python
    def _render_library_rows(self) -> None:
        query = self.state.search_query.lower()
        records = self._characters
        if query:
            if len(self._characters) >= self.LIBRARY_FTS_THRESHOLD:
                from ..CCP_Modules.ccp_character_handler import search_characters_fts
                records = search_characters_fts(query)
            else:
                records = [
                    record for record in self._characters
                    if query in str(record.get("name", "")).lower()
                ]
        rows = tuple(
            LibraryRow(
                item_id=str(record.get("id", "")),
                kind="character",
                name=str(record.get("name", "Unnamed")),
            )
            for record in records
        )
        self.query_one(PersonasLibraryPane).update_rows(
            rows,
            total=len(self._characters),
            noun=MODE_NOUNS["characters"],
            filtered=bool(query),
        )
```

Add the personas-mode equivalent filtering `self._profiles` by name (local only — the
profile list is small), reusing the same `filtered=bool(query)` count style:

```python
    def _render_profile_rows_filtered(self) -> None:
        query = self.state.search_query.lower()
        records = [
            record for record in self._profiles
            if not query or query in str(record.get("name", "")).lower()
        ]
        rows = tuple(
            LibraryRow(
                item_id=str(record.get("id", "")),
                kind="persona_profile",
                name=str(record.get("name", "Unnamed")),
            )
            for record in records
        )
        self.query_one(PersonasLibraryPane).update_rows(
            rows, total=len(self._profiles), noun=MODE_NOUNS["personas"],
            filtered=bool(query),
        )
```

(Refactor `_refresh_persona_rows` to call `_render_profile_rows_filtered()` after
storing `self._profiles` so both paths render identically.)

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py \
        tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py \
        Tests/UI/test_personas_workbench.py
git commit -m "feat: add Personas workbench library search with FTS fallback

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 10: Import and export

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Reference: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py:1617`
  (`import_and_save_character_from_file`), `:2666` (`export_character_card_to_json`),
  `:2743` (`export_character_card_to_png`); legacy import flow in
  `tldw_chatbook/UI/Screens/ccp_screen.py` (search for `ccp-import-character-native`)

- [ ] **Step 1: Write the failing tests** (new class)

```python
class TestImportExport:
    async def test_import_success_refreshes_and_selects(self, mock_app_instance, stub_characters, monkeypatch, tmp_path):
        card = tmp_path / "card.json"
        card.write_text('{"name": "Imported Hero"}')
        imported = {"id": 3, "name": "Imported Hero", "version": 1}
        monkeypatch.setattr(
            character_handler_module, "import_character_card", lambda path: imported
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await screen._import_character_from_path(str(card))
            await pilot.pause()
            assert screen.state.selected_id == "3"

    async def test_import_failure_shows_recovery_copy(self, mock_app_instance, stub_characters, monkeypatch, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not a card")
        def boom(path):
            raise ValueError("Unsupported card format")
        monkeypatch.setattr(character_handler_module, "import_character_card", boom)
        notifications = []
        mock_app_instance.notify = lambda msg, severity="warning": notifications.append(msg)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._import_character_from_path(str(bad))
            await pilot.pause()
        assert any("Unsupported card format" in str(n) for n in notifications)

    async def test_export_json_writes_file(self, mock_app_instance, stub_characters, monkeypatch, tmp_path):
        import tldw_chatbook.UI.Screens.personas_screen as screen_module
        monkeypatch.setattr(
            screen_module, "export_character_card_to_json",
            lambda db, character_id, **kwargs: '{"name": "Detective Sam"}',
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            target = tmp_path / "sam.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
        assert target.read_text() == '{"name": "Detective Sam"}'
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q "Tests/UI/test_personas_workbench.py::TestImportExport" --tb=short`
Expected: FAIL — methods missing

- [ ] **Step 3: Implement**

In `personas_screen.py`, add imports:

```python
from ...Character_Chat.Character_Chat_Lib import (
    export_character_card_to_json,
    export_character_card_to_png,
)
```

Add handlers and helpers (the file-picker plumbing copies the legacy pattern — see the
`ccp-import-character-native` handler in `ccp_screen.py` for the dialog it opens; reuse
the same dialog class):

```python
    @on(LibraryImportRequested)
    async def _import_requested(self, message: LibraryImportRequested) -> None:
        if self.state.active_mode != "characters":
            return
        # Reuse the file-open dialog the legacy route uses for character import
        # (see ccp_screen.py #ccp-import-character-native handler); on selection it
        # should call self._import_character_from_path(path).
        await self._open_character_import_dialog()

    async def _import_character_from_path(self, path: str) -> None:
        from ..CCP_Modules import ccp_character_handler as handler_module
        try:
            imported = handler_module.import_character_card(path)
        except Exception as exc:
            logger.warning(f"Character import failed: {exc}")
            self._notify(f"Import failed: {exc}", "error")
            return
        if not imported:
            self._notify("Import failed: file did not contain a valid character card.", "error")
            return
        await self.character_handler.refresh_character_list()
        new_id = str(imported.get("id", "")) if isinstance(imported, Mapping) else ""
        if new_id:
            from dataclasses import replace
            self.state = replace(
                self.state, selected_kind="character", selected_id=new_id,
                edit_mode="view", is_unsaved=False,
            )
            await self.character_handler.load_character(new_id)
            self._show_center("#ccp-character-card-view")
            record = self._character_record(new_id)
            self.query_one(PersonasInspectorPane).show_selection(
                name=str((record or {}).get("name", "Imported character")),
                kind="character",
                authority="Local",
            )
        self._notify("Character imported.", "information")

    async def _export_selected_character(self, target_path: str, *, fmt: str) -> None:
        if self.state.selected_kind != "character" or self.state.selected_id is None:
            return
        db = self._character_db()
        try:
            if fmt == "json":
                payload = export_character_card_to_json(db, int(self.state.selected_id))
                if payload is None:
                    raise ValueError("export returned no data")
                with open(target_path, "w", encoding="utf-8") as handle:
                    handle.write(payload)
            else:
                export_character_card_to_png(db, int(self.state.selected_id), target_path)
        except Exception as exc:
            logger.warning(f"Character export failed: {exc}")
            self._notify(f"Export failed: {exc}", "error")
            return
        self._notify(f"Exported to {target_path}", "information")

    def _character_db(self):
        from ..CCP_Modules.ccp_character_handler import _default_character_db
        return _default_character_db()
```

Wire `#personas-export-json` / `#personas-export-png` button presses to open a
save-file dialog (same dialog family the legacy export used) and then call
`_export_selected_character(path, fmt=...)`. Check `export_character_card_to_png`'s
exact parameter order at `Character_Chat_Lib.py:2743` before wiring — it takes the
output path; match its signature.

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat: wire character import and JSON/PNG export in Personas workbench

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 11: Saved conversations panel and read-only view

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Reference: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py:2139`
  (`list_character_conversations(db, character_id, limit, offset)`),
  `tldw_chatbook/Widgets/CCP_Widgets/ccp_conversation_view_widget.py`

- [ ] **Step 1: Write the failing tests** (new class)

```python
class TestConversationsPanel:
    async def test_selecting_character_lists_conversations(self, mock_app_instance, stub_characters, monkeypatch):
        import tldw_chatbook.UI.Screens.personas_screen as screen_module
        monkeypatch.setattr(
            screen_module, "list_character_conversations",
            lambda db, character_id, limit=20, offset=0: [
                {"id": "conv-1", "title": "First case"},
            ],
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.pause()
            rows = screen.query(".personas-conversation-row")
            assert [str(r.label) for r in rows] == ["First case"]

    async def test_conversation_row_opens_readonly_view(self, mock_app_instance, stub_characters, monkeypatch):
        import tldw_chatbook.UI.Screens.personas_screen as screen_module
        monkeypatch.setattr(
            screen_module, "list_character_conversations",
            lambda db, character_id, limit=20, offset=0: [
                {"id": "conv-1", "title": "First case"},
            ],
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#personas-conversation-row-0")
            await pilot.pause()
            view = screen.query_one("#ccp-conversation-messages-view")
            assert view.display is True
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q "Tests/UI/test_personas_workbench.py::TestConversationsPanel" --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement**

In `personas_screen.py`:

1. Imports:

```python
from ...Character_Chat.Character_Chat_Lib import list_character_conversations
from ...Widgets.CCP_Widgets import CCPConversationViewWidget
from ...Widgets.Persona_Widgets.personas_messages import ConversationRowSelected
```

2. Mount `CCPConversationViewWidget(parent_screen=self)` in `#personas-detail-stack`
and add its view id (`#ccp-conversation-messages-view` — confirm the widget's own id by
reading `ccp_conversation_view_widget.py` compose; use what it actually sets) to
`_show_center`.

3. After a character selection succeeds in `_row_selected`, load conversations in a
worker:

```python
            self._load_character_conversations(message.item_id)
```

```python
    @work(thread=True, exclusive=True)
    def _load_character_conversations(self, character_id: str) -> None:
        try:
            conversations = list_character_conversations(
                self._character_db(), int(character_id), limit=20
            )
        except Exception as exc:
            logger.warning(f"Conversation list failed: {exc}")
            conversations = []
        rows = tuple(
            (str(c.get("id", "")), str(c.get("title") or "Untitled conversation"))
            for c in conversations
        )
        self.app.call_from_thread(
            self.query_one(PersonasInspectorPane).show_conversations, rows
        )
```

4. Open a selected conversation read-only via the conversation handler path the
legacy screen used (`CCPConversationHandler` exists; instantiate it in `__init__` like
the other handlers if its load API is needed, or drive
`CCPConversationViewWidget` directly if it exposes a load method — read the widget
first and pick the one that already works):

```python
    @on(ConversationRowSelected)
    async def _conversation_selected(self, message: ConversationRowSelected) -> None:
        self._show_center("#ccp-conversation-messages-view")
        await self._load_conversation_messages(message.conversation_id)
```

`_load_conversation_messages` should call
`Character_Chat_Lib.retrieve_conversation_messages_for_ui` (line 2435) in a thread
worker and hand the result to the conversation view widget's existing population
method. Add two inspector follow-up actions inside the conversation view footer:
**Continue in Console** posts a `ChatHandoffPayload` (same construction as Task 12's
attach, plus `metadata["conversation_id"]`) and **Open in Library** posts
`NavigateToScreen("conversation")` (import `NavigateToScreen` from
`..Navigation.main_navigation`).

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat: add saved-conversations panel and read-only view to Personas workbench

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 12: Attach to Console and Start Chat

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Reference: attach handler in the pre-rewrite `personas_screen.py`
  (`git show HEAD~N:tldw_chatbook/UI/Screens/personas_screen.py`, the
  `attach_to_console` method) and `ccp_screen.py` `#ccp-attach-selected-to-console` /
  `#ccp-start-selected-chat` handlers

- [ ] **Step 1: Write the failing tests** (new class)

```python
class TestConsoleActions:
    async def test_attach_stages_selected_character_payload(self, mock_app_instance, stub_characters):
        staged = []
        mock_app_instance.open_chat_with_handoff = lambda payload: staged.append(payload)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.click("#personas-attach-to-console")
            await pilot.pause()
        assert len(staged) == 1
        payload = staged[0]
        assert payload.source == "personas"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"
        assert "Detective Sam" in payload.title

    async def test_start_chat_passes_start_intent(self, mock_app_instance, stub_characters):
        staged = []
        mock_app_instance.open_chat_with_handoff = lambda payload: staged.append(payload)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.click("#personas-start-chat")
            await pilot.pause()
        assert staged[0].metadata["intent"] == "start_chat"
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q "Tests/UI/test_personas_workbench.py::TestConsoleActions" --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
    @on(Button.Pressed, "#personas-attach-to-console")
    def _attach_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._stage_console_handoff(intent="attach")

    @on(Button.Pressed, "#personas-start-chat")
    def _start_chat_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._stage_console_handoff(intent="start_chat")

    def _stage_console_handoff(self, *, intent: str) -> None:
        if self.state.selected_id is None or self.state.selected_kind is None:
            self._notify("Select a character or persona first.", "warning")
            return
        if self.state.selected_kind == "character":
            record = self._character_record(self.state.selected_id) or {}
            target_type = "character"
        else:
            record = self._profile_record(self.state.selected_id) or {}
            target_type = "persona_profile"
        name = str(record.get("name", "Unnamed"))
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            self._notify("Console handoff is unavailable in this runtime.", "warning")
            return
        body_lines = [f"{target_type}: {name}"]
        for key in ("description", "personality", "system_prompt"):
            value = str(record.get(key, "") or "").strip()
            if value:
                body_lines.append(f"{key}: {value}")
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="personas",
                item_type=f"{target_type}-card",
                title=f"{name} ({target_type})",
                body="\n".join(body_lines),
                display_summary=f"{name} staged from Personas.",
                suggested_prompt=(
                    f"Respond as {name}." if intent == "start_chat"
                    else f"Use {name} to guide the next response."
                ),
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata={
                    "intent": intent,
                    "selected_kind": target_type,
                    "selected_record_id": self.state.selected_id,
                    "selected_target_id": f"local:{target_type}:{self.state.selected_id}",
                    "backend": "local",
                },
            )
        )
```

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat: wire Attach to Console and Start Chat from Personas inspector

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 13: Preview conversation pane

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_preview.py`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_personas_preview.py
"""Tests for the ephemeral Personas preview conversation."""

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
    PersonasPreviewPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PreviewReplyRequested,
)

pytestmark = pytest.mark.asyncio


class PreviewApp(App):
    def compose(self):
        yield PersonasPreviewPane(id="personas-preview-pane")


async def test_collapsed_by_default_and_toggle_expands():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        body = pilot.app.query_one("#personas-preview-body")
        assert body.display is False
        await pilot.click("#personas-preview-toggle")
        await pilot.pause()
        assert body.display is True


async def test_seed_greeting_and_reset():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        pane.seed_greeting("Hello, I am Detective Sam.")
        await pilot.pause()
        transcript = pilot.app.query_one("#personas-preview-transcript")
        assert len(transcript.children) == 1
        pane.append_user("Who are you?")
        pane.append_reply("A detective.")
        await pilot.pause()
        assert len(transcript.children) == 3
        pane.reset()
        await pilot.pause()
        assert len(transcript.children) == 1  # greeting only


async def test_test_reply_posts_message_with_input_text():
    received = []

    class CaptureApp(PreviewApp):
        def on_preview_reply_requested(self, message: PreviewReplyRequested) -> None:
            received.append(message.user_message)

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        pilot.app.query_one("#personas-preview-input", Input).value = "Who did it?"
        await pilot.click("#personas-preview-test-reply")
        await pilot.pause()
    assert received == ["Who did it?"]


async def test_provider_status_is_readable():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.set_status("Provider unavailable - configure in Settings")
        await pilot.pause()
        status = pilot.app.query_one("#personas-preview-status", Static)
        assert "Provider unavailable" in str(status.renderable)
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q Tests/UI/test_personas_preview.py --tb=short`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the pane**

```python
# tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py
"""Ephemeral preview conversation pane. Never persists anything."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from .personas_messages import (
    PreviewOpenInConsoleRequested,
    PreviewReplyRequested,
    PreviewResetRequested,
)


class PersonasPreviewPane(Vertical):
    """Greeting + ephemeral test exchanges with the in-editor draft."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._greeting: str | None = None

    def compose(self) -> ComposeResult:
        yield Button(
            "Preview conversation",
            id="personas-preview-toggle",
            tooltip="Try the selected character's voice without saving anything.",
        )
        with Vertical(id="personas-preview-body"):
            yield Static("", id="personas-preview-status")
            yield VerticalScroll(id="personas-preview-transcript")
            yield Input(placeholder="Test message...", id="personas-preview-input")
            with Horizontal(classes="ds-toolbar"):
                yield Button("Test Reply", id="personas-preview-test-reply")
                yield Button("Reset", id="personas-preview-reset")
                yield Button("Open in Console", id="personas-preview-open-console")

    def on_mount(self) -> None:
        self.query_one("#personas-preview-body").display = False

    def expand(self) -> None:
        self.query_one("#personas-preview-body").display = True

    def seed_greeting(self, greeting: str) -> None:
        self._greeting = greeting
        self._clear_transcript()
        if greeting:
            self._mount_line("character", greeting)

    def append_user(self, text: str) -> None:
        self._mount_line("you", text)

    def append_reply(self, text: str) -> None:
        self._mount_line("character", text)

    def reset(self) -> None:
        self.seed_greeting(self._greeting or "")
        self.set_status("")

    def set_status(self, text: str) -> None:
        self.query_one("#personas-preview-status", Static).update(text)

    def transcript_text(self) -> str:
        lines = []
        for child in self.query_one("#personas-preview-transcript").children:
            if isinstance(child, Static):
                lines.append(str(child.renderable))
        return "\n".join(lines)

    def _clear_transcript(self) -> None:
        self.query_one("#personas-preview-transcript", VerticalScroll).remove_children()

    def _mount_line(self, speaker: str, text: str) -> None:
        self.query_one("#personas-preview-transcript", VerticalScroll).mount(
            Static(f"{speaker}: {text}", classes=f"personas-preview-line-{speaker}")
        )

    @on(Button.Pressed, "#personas-preview-toggle")
    def _toggle(self, event: Button.Pressed) -> None:
        event.stop()
        body = self.query_one("#personas-preview-body")
        body.display = not body.display

    @on(Button.Pressed, "#personas-preview-test-reply")
    def _test_reply(self, event: Button.Pressed) -> None:
        event.stop()
        field = self.query_one("#personas-preview-input", Input)
        text = field.value.strip()
        if not text:
            return
        field.value = ""
        self.append_user(text)
        self.post_message(PreviewReplyRequested(user_message=text))

    @on(Button.Pressed, "#personas-preview-reset")
    def _reset_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.reset()
        self.post_message(PreviewResetRequested())

    @on(Button.Pressed, "#personas-preview-open-console")
    def _open_console(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PreviewOpenInConsoleRequested())
```

- [ ] **Step 4: Verify pane green**

Run: `python -m pytest -q Tests/UI/test_personas_preview.py --tb=short`
Expected: 4 passed

- [ ] **Step 5: Wire the screen to the provider gateway**

In `personas_screen.py`:

1. Mount `PersonasPreviewPane(id="personas-preview-pane")` at the bottom of
`#personas-work-area` (after `#personas-detail-stack`).
2. On character selection (`_row_selected` character branch), seed the greeting:

```python
            from ...Character_Chat.Character_Chat_Lib import replace_placeholders
            preview = self.query_one(PersonasPreviewPane)
            greeting = replace_placeholders(
                str((record or {}).get("first_message", "") or ""),
                str((record or {}).get("name", "")),
                "User",
            )
            preview.seed_greeting(greeting)
```

3. Add the gateway-backed reply worker:

```python
from ...Chat.console_chat_models import ConsoleProviderSelection
from ...Chat.console_provider_gateway import ConsoleProviderGateway
from ...Widgets.Persona_Widgets.personas_messages import (
    PreviewOpenInConsoleRequested,
    PreviewReplyRequested,
)
```

```python
    _preview_history: list[dict[str, str]]
    _preview_gateway: ConsoleProviderGateway | None = None

    def _ensure_preview_gateway(self) -> ConsoleProviderGateway:
        if self._preview_gateway is None:
            self._preview_gateway = ConsoleProviderGateway(
                config_provider=lambda: getattr(self.app_instance, "app_config", {}) or {},
            )
        return self._preview_gateway

    def _preview_system_prompt(self) -> str:
        record = (
            self._character_record(self.state.selected_id or "")
            if self.state.selected_kind == "character"
            else self._profile_record(self.state.selected_id or "")
        ) or {}
        parts = [str(record.get(k, "") or "") for k in
                 ("system_prompt", "personality", "description", "scenario")]
        return "\n".join(p for p in parts if p).strip() or "Stay in character."

    @on(PreviewReplyRequested)
    def _preview_reply(self, message: PreviewReplyRequested) -> None:
        self._preview_history.append({"role": "user", "content": message.user_message})
        self._run_preview_reply()

    @work(exclusive=True)
    async def _run_preview_reply(self) -> None:
        pane = self.query_one(PersonasPreviewPane)
        config = getattr(self.app_instance, "app_config", {}) or {}
        defaults = config.get("character_defaults", {}) if isinstance(config, dict) else {}
        provider = str(defaults.get("provider", "") or "")
        model = str(defaults.get("model", "") or "")
        gateway = self._ensure_preview_gateway()
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(provider=provider, explicit_model=model or None)
        )
        if not resolution.ready:
            pane.set_status(
                resolution.visible_copy or "Provider unavailable - configure in Settings"
            )
            return
        pane.set_status("Running")
        messages = [{"role": "system", "content": self._preview_system_prompt()}]
        messages.extend(self._preview_history)
        chunks: list[str] = []
        try:
            async for chunk in gateway.stream_chat(resolution, messages):
                chunks.append(chunk)
        except Exception as exc:
            logger.warning(f"Preview reply failed: {exc}")
            pane.set_status("Provider error - try again or configure in Settings")
            return
        reply = "".join(chunks).strip()
        if reply:
            self._preview_history.append({"role": "assistant", "content": reply})
            pane.append_reply(reply)
        pane.set_status("Ready")

    @on(PreviewOpenInConsoleRequested)
    def _preview_open_console(self, message: PreviewOpenInConsoleRequested) -> None:
        pane = self.query_one(PersonasPreviewPane)
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            self._notify("Console handoff is unavailable in this runtime.", "warning")
            return
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="personas",
                item_type="preview-conversation",
                title="Personas preview conversation",
                body=pane.transcript_text(),
                display_summary="Preview conversation staged from Personas.",
                suggested_prompt="Continue this conversation in character.",
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata={"selected_kind": self.state.selected_kind or "",
                          "selected_record_id": self.state.selected_id or ""},
            )
        )
```

Initialize `self._preview_history = []` in `__init__`; clear it in `seed` points
(row selection and `PreviewResetRequested` handler:
`@on(PreviewResetRequested)` → `self._preview_history.clear()`). Close the gateway in
`on_unmount` with `self.run_worker(self._preview_gateway.aclose())` guarded for None.
Note the spec point: the preview reads the in-editor draft — when
`self.state.edit_mode in ("edit", "create")`, build `_preview_system_prompt` from
`self.query_one(CCPCharacterEditorWidget)`'s current field values instead of the saved
record (the editor exposes the same data it posts in `CharacterSaveRequested`; read
`ccp_character_editor_widget.py` for its collect method and use it).

- [ ] **Step 6: Add a screen-level test** (append to `Tests/UI/test_personas_workbench.py`)

```python
class TestPreviewIntegration:
    async def test_preview_blocked_provider_shows_readable_status(self, mock_app_instance, stub_characters, monkeypatch):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
                PreviewReplyRequested,
            )
            screen.post_message(PreviewReplyRequested(user_message="Hello"))
            await pilot.pause()
            await pilot.pause()
            status = screen.query_one("#personas-preview-status", Static)
            text = str(status.renderable)
            assert text  # readable status, not a traceback
            assert "Traceback" not in text
```

- [ ] **Step 7: Verify all green**

Run: `python -m pytest -q Tests/UI/test_personas_preview.py Tests/UI/test_personas_workbench.py --tb=short`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py \
        tldw_chatbook/UI/Screens/personas_screen.py \
        Tests/UI/test_personas_preview.py Tests/UI/test_personas_workbench.py
git commit -m "feat: add ephemeral preview conversation to Personas workbench

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 14: Delete with confirmation

**Files:**
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py` (module-level delete)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Write the failing test**

```python
class TestDelete:
    async def test_delete_soft_deletes_and_refreshes(self, mock_app_instance, stub_characters, monkeypatch):
        deleted = []
        monkeypatch.setattr(
            character_handler_module, "delete_character",
            lambda character_id, expected_version: deleted.append(character_id) or True,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            screen._confirm_delete = lambda name: _async_true()
            await pilot.click("#personas-delete")
            await pilot.pause()
            await pilot.pause()
        assert deleted == ["1"]


async def _async_true():
    return True
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest -q "Tests/UI/test_personas_workbench.py::TestDelete" --tb=short`
Expected: FAIL — `delete_character` does not exist

- [ ] **Step 3: Implement**

Add to `ccp_character_handler.py` (module level, next to `update_character`):

```python
def delete_character(character_id: CharacterId, expected_version: int) -> bool:
    """Soft-delete a character card with optimistic locking."""
    db = _default_character_db()
    if db is None:
        return False
    return bool(
        db.soft_delete_character_card(int(character_id), expected_version)
    )
```

In `personas_screen.py`:

```python
    @on(Button.Pressed, "#personas-delete")
    async def _delete_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.state.selected_id is None:
            return
        if self.state.selected_kind == "character":
            record = self._character_record(self.state.selected_id) or {}
            name = str(record.get("name", "this character"))
            if not await self._confirm_delete(name):
                return
            from ..CCP_Modules import ccp_character_handler as handler_module
            try:
                ok = handler_module.delete_character(
                    self.state.selected_id, int(record.get("version", 1))
                )
            except Exception as exc:
                self._notify(f"Delete failed: {exc}", "error")
                return
            if not ok:
                self._notify(
                    "Delete failed: the character changed since it was loaded. "
                    "Reload and try again.",
                    "error",
                )
                return
            await self._after_delete()
        elif self.state.selected_kind == "persona_profile":
            record = self._profile_record(self.state.selected_id) or {}
            name = str(record.get("name", "this persona"))
            if not await self._confirm_delete(name):
                return
            service = getattr(self.app_instance, "character_persona_scope_service", None)
            try:
                await service.delete_persona_profile(self.state.selected_id, mode="local")
            except Exception as exc:
                self._notify(f"Delete failed: {exc}", "error")
                return
            await self._after_delete()

    async def _confirm_delete(self, name: str) -> bool:
        from ...Widgets.confirmation_dialog import ConfirmationDialog
        return bool(
            await self.app.push_screen_wait(
                ConfirmationDialog(message=f"Delete {name}? This cannot be undone here.")
            )
        )

    async def _after_delete(self) -> None:
        self.state = self.state.with_mode(self.state.active_mode)
        self.query_one(PersonasInspectorPane).clear_selection()
        self._show_center(None)
        if self.state.active_mode == "characters":
            await self.character_handler.refresh_character_list()
        else:
            self._refresh_persona_rows()
        self._notify("Deleted.", "information")
```

Check `ConfirmationDialog.__init__` at `confirmation_dialog.py:70` and
`delete_persona_profile`'s signature at `character_persona_scope_service.py:684` (it
may also need an expected version argument) before wiring; match what exists.

- [ ] **Step 4: Verify green**

Run: `python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py \
        tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat: add confirmed delete for characters and persona profiles

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Phase 3 exit criteria:** all spec workflows pass; `python -m pytest -q Tests/UI --tb=short` green.
Capture a textual-web/CDP screenshot of the workbench (Characters mode with a selection,
Personas mode) under `Docs/superpowers/qa/personas-workbench/` and present for approval
before Phase 4.

---

## Phase 4 — Route flip and legacy retirement

### Task 15: Flip route resolution to the personas destination

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py:78`
- Test: `Tests/UI/test_shell_destinations.py` (or create if missing — check
  `grep -rl resolve_shell_route Tests/` first and extend the existing file)

- [ ] **Step 1: Write the failing tests**

```python
def test_ccp_legacy_routes_resolve_to_personas_destination():
    from tldw_chatbook.UI.Navigation.shell_destinations import resolve_shell_route

    for legacy in ("ccp", "characters", "prompts", "conversations_characters_prompts"):
        resolved = resolve_shell_route(legacy)
        assert resolved.destination_id == "personas"
        assert resolved.canonical_route == "personas"


def test_ccp_screen_route_loads_personas_screen():
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target

    screen_name, _tab, screen_class = resolve_screen_target("ccp")
    assert screen_class is not None
    assert screen_class.__name__ == "PersonasScreen"
```

- [ ] **Step 2: Verify red**

Run the test file.
Expected: FAIL — `ccp` still resolves to itself / `ConversationScreen`

- [ ] **Step 3: Implement**

In `shell_destinations.py`:
- Remove `"ccp"` from `_ROUTABLE_LEGACY_ROUTES` (line ~144).
- Delete the `"conversations_characters_prompts"`, `"characters"`, and `"prompts"`
  entries from `_CANONICAL_ROUTE_OVERRIDES` (keep `"subscription"`). With `ccp` no
  longer routable, all four legacy routes fall through to the destination's
  `primary_route` (`personas`).

In `screen_registry.py` line 78, change:

```python
    "ccp": ScreenRoute("ccp", TAB_CCP, "tldw_chatbook.UI.Screens.conversation_screen", "ConversationScreen"),
```

to:

```python
    "ccp": ScreenRoute("ccp", "personas", "tldw_chatbook.UI.Screens.personas_screen", "PersonasScreen"),
```

- [ ] **Step 4: Verify green and run navigation suites**

Run: `python -m pytest -q Tests/UI/test_shell_destinations.py Tests/UI --tb=short`
(adjust the first path to wherever the route tests live)
Expected: route tests pass; any test asserting the old mapping gets updated to the new
expectation in this commit.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Navigation/ Tests/
git commit -m "feat: resolve ccp/characters/prompts legacy routes to the Personas workbench

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 16: Retire the legacy CCP screen and sidebar chrome

**Files:**
- Delete: `tldw_chatbook/UI/Screens/ccp_screen.py`, `tldw_chatbook/UI/Screens/ccp_screen.py.bak`,
  `tldw_chatbook/UI/Screens/conversation_screen.py`,
  `tldw_chatbook/Widgets/CCP_Widgets/ccp_sidebar_widget.py`,
  `tldw_chatbook/UI/CCP_Modules/ccp_sidebar_handler.py`
- Modify: `tldw_chatbook/Widgets/CCP_Widgets/__init__.py`,
  `tldw_chatbook/UI/CCP_Modules/__init__.py`
- Modify/Delete: `Tests/UI/test_ccp_screen.py`

- [ ] **Step 1: Find every remaining importer**

```bash
grep -rn "ccp_screen\|conversation_screen\|ccp_sidebar\|CCPScreen\|ConversationScreen" \
  tldw_chatbook Tests --include="*.py" | grep -v __pycache__
```

Expected: hits only in the files being deleted, the two `__init__.py` files, and
`Tests/UI/test_ccp_screen.py`. Any other hit (e.g. `app.py`, `Event_Handlers/`) must be
fixed in this task before deleting — replace screen-class references with
`PersonasScreen` or remove dead branches, smallest change that keeps imports clean.

- [ ] **Step 2: Port surviving test intent**

Review `Tests/UI/test_ccp_screen.py`. Behavior already covered by
`test_personas_workbench.py` (selection, save, attach, mode switching) is deleted with
the file. Anything still unique (e.g. import-flow edge cases) moves into
`test_personas_workbench.py` rewritten against the new IDs. `test_ccp_handlers.py` and
`test_ccp_prompt_handler_scope.py` test reused modules — keep them; remove only tests
of `ccp_sidebar_handler`.

- [ ] **Step 3: Delete and clean exports**

```bash
git rm tldw_chatbook/UI/Screens/ccp_screen.py tldw_chatbook/UI/Screens/ccp_screen.py.bak \
       tldw_chatbook/UI/Screens/conversation_screen.py \
       tldw_chatbook/Widgets/CCP_Widgets/ccp_sidebar_widget.py \
       tldw_chatbook/UI/CCP_Modules/ccp_sidebar_handler.py \
       Tests/UI/test_ccp_screen.py
```

Remove the corresponding names from `CCP_Widgets/__init__.py` and
`CCP_Modules/__init__.py`.

- [ ] **Step 4: Verify the app imports and the suite passes**

```bash
python -c "import tldw_chatbook.app"
python -m pytest -q Tests/UI Tests/Chat --tb=short
```

Expected: import succeeds, suites pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: retire legacy CCP screen, shim, and sidebar chrome

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 17: Docs, QA evidence, and task closure

**Files:**
- Modify: `Docs/Design/master-shell-route-inventory.md` (ccp/personas rows)
- Add: screenshots under `Docs/superpowers/qa/personas-workbench/`
- Modify: the backlog task from Task 1

- [ ] **Step 1: Update the route inventory** — mark `ccp`, `characters`, `prompts` as
  resolving to the `personas` destination; note the workbench replaces the snapshot shell.

- [ ] **Step 2: Capture actual screenshots** via the established textual-web/CDP
  workflow: Characters mode with a selected character (card + conversations + preview
  expanded), the editor with a validation error, and Personas mode. Save under
  `Docs/superpowers/qa/personas-workbench/` and present for approval.

- [ ] **Step 3: Run the full test suite**

```bash
python -m pytest -q --tb=short
```

Expected: green (pre-existing unrelated failures noted explicitly if any).

- [ ] **Step 4: Close out the backlog task**

```bash
backlog task edit <task-N> -s Done --notes "Personas workbench shipped per spec \
Docs/superpowers/specs/2026-06-09-personas-workbench-design.md and ADR-004; legacy \
ccp route/screen retired; all ACs verified by Tests/UI/test_personas_workbench.py \
and screenshot QA."
```

Check all ACs (`- [ ]` → `- [x]`) in the task file and add Implementation Notes per the
repo's Definition of Done.

- [ ] **Step 5: Commit**

```bash
git add Docs/ backlog/
git commit -m "docs: update route inventory and QA evidence for Personas workbench

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review checklist (run after writing, fixed inline)

- Spec coverage: layout/panes (Tasks 3-5), personas mode (7-8), search (9),
  import/export (10), conversations (11), Console actions (12), preview (13),
  delete (14), route flip (15), retirement (16), states/shortcuts (4, 6), ADR (1),
  QA/docs (17). Optimistic-lock conflict surfaces in Task 14's version-mismatch path;
  the recovery-callout variant for service/policy failures reuses the notify path plus
  inspector copy and is asserted in Task 13's readable-status test.
- Verify-before-wiring notes are deliberate (dialog signatures, PNG export parameter
  order, conversation view ID, scope-service delete signature): those APIs exist but
  their exact shapes must be read at implementation time rather than guessed here.
- Type consistency: `LibraryRow(item_id, kind, name, is_unsaved)`,
  `update_rows(rows, *, total, noun, filtered=False)`, `show_selection(*, name, kind,
  authority)`, `PersonasWorkbenchState.with_mode`, message constructor arguments —
  used identically across tasks.
```
