# Console Conversation Rail Overflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Console workspace Conversations subsection bounded, collapsible, and searchable so many conversations cannot hide lower Context rail content.

**Architecture:** Keep `ChatScreen` as the owner of interaction state, service calls, and persistence. Add a conversation-section display state beside the existing workspace display state, render it in `ConsoleWorkspaceContextTray`, and keep search/collapse behavior scoped to the active workspace without changing workspace storage, chat storage, sync, or handoff contracts.

**Tech Stack:** Python 3.11+, Textual, TCSS, existing Console widgets, existing workspace registry and chat conversation scope services, pytest mounted UI tests, Textual-web/CDP visual QA.

---

## Source Material

- Backlog task: `backlog/tasks/task-138 - Fix-Console-conversation-rail-overflow.md`
- Approved design: `Docs/superpowers/specs/2026-06-25-console-conversation-rail-overflow-design.md`
- Governing ADR: `backlog/decisions/005-console-workspace-server-readiness.md`
- Current tray widget: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Current Console screen: `tldw_chatbook/UI/Screens/chat_screen.py`
- Workspace display state: `tldw_chatbook/Workspaces/display_state.py`
- Workspace registry service: `tldw_chatbook/Workspaces/registry_service.py`
- Chat scope service: `tldw_chatbook/Chat/chat_conversation_scope_service.py`
- Existing mounted tests:
  - `Tests/UI/test_console_workspace_context_rail.py`
  - `Tests/UI/test_console_native_chat_flow.py`
  - `Tests/UI/test_console_session_settings.py`

## Scope Check

This is one bounded UI/workflow slice:

- In scope: Console left Context rail Conversations subsection layout, per-workspace collapse preference, transient active-workspace search, result cap copy, row selection with active query preserved, mounted tests, screenshot evidence, Backlog task hygiene.
- Out of scope: schema migrations, workspace ownership changes, server hydration, sync, Library-only browsing, advanced filters, bulk actions, deletion, pagination/infinite scroll.

ADR required: no
ADR path: N/A
Reason: this changes Console presentation and UI preference state only. It does not change workspace ownership, storage/schema, sync policy, provider/runtime boundary, server handoff contract, or data ownership.

## File Structure

- Modify `backlog/tasks/task-138 - Fix-Console-conversation-rail-overflow.md`
  - Add implementation plan before code execution.
  - Check ACs and add implementation notes after implementation.

- Modify `tldw_chatbook/Workspaces/display_state.py`
  - Add `ConsoleWorkspaceConversationSectionState`.
  - Add conversation-section constants for limit and adaptive sizing.
  - Add pure helpers for selected summary, count/status copy, and adaptive visible-row calculation.
  - Extend `ConsoleWorkspaceContextState` with optional `conversation_section`.

- Modify `tldw_chatbook/Widgets/Console/console_workspace_context.py`
  - Render header/count, selected summary, collapse/expand toggle, search input, clear action, bounded `VerticalScroll` row list, and expanded `New conversation`.
  - Preserve existing conversation row IDs/classes and `conversation_id` attributes.
  - Keep row labels markup-safe.

- Modify `tldw_chatbook/UI/Screens/chat_screen.py`
  - Own transient search query, debounce timer, stale-result token, and per-workspace collapse preference.
  - Build default section from native sessions plus workspace memberships.
  - Build search section from native sessions, filtered memberships, and persisted workspace-scoped conversation search.
  - Handle search input, clear, collapse toggle, and resize sync.

- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add compact styles for the conversation subsection controls and bounded list.

- Modify `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Mirror the Console TCSS rules or regenerate it with the repo CSS build script if available in the worker environment.

- Modify `Tests/UI/test_console_workspace_context_rail.py`
  - Add pure/display-state and mounted tray regressions for adaptive bound, collapse rendering, and lower content reachability.

- Modify `Tests/UI/test_console_native_chat_flow.py`
  - Add mounted workflow regressions for search scope, stale search guard, result selection, search cap/error/empty copy, and workspace switch behavior.

---

### Task 0: Backlog Task Hygiene

**Files:**
- Modify: `backlog/tasks/task-138 - Fix-Console-conversation-rail-overflow.md`

- [ ] **Step 1: Move the task to In Progress and add the implementation plan summary**

Run:

```bash
backlog task edit 134 -s "In Progress" --plan "1. Add Console workspace conversation-section display state and pure tests.
2. Render bounded/collapsible/searchable Conversations UI in ConsoleWorkspaceContextTray.
3. Add ChatScreen-owned query, collapse preference, search, stale-result, and row-selection wiring.
4. Add TCSS rules for bounded list, summary, search, and collapse controls.
5. Add mounted workflow regressions for overflow, collapse, search scope, stale results, selection, and workspace switching.
6. Run focused Console tests, git diff --check, capture Textual-web/CDP evidence, then update TASK-138 notes.

ADR required: no
ADR path: N/A
Reason: presentation and UI preference state only; no schema, sync, workspace ownership, provider/runtime, or handoff contract change."
```

Expected: `TASK-138` status becomes `In Progress` and contains the ADR check in its Implementation Plan section.

- [ ] **Step 2: Confirm the task file contains the plan**

Run:

```bash
backlog task 134 --plain
```

Expected: output includes `ADR required: no` and the six implementation steps.

- [ ] **Step 3: Commit the task status/plan update**

Run:

```bash
git add "backlog/tasks/task-138 - Fix-Console-conversation-rail-overflow.md"
git commit -m "Track Console conversation rail overflow task"
```

Expected: commit succeeds with only the task file staged.

---

### Task 1: Add Conversation Section Display State

**Files:**
- Modify: `tldw_chatbook/Workspaces/display_state.py`
- Test: `Tests/UI/test_console_workspace_context_rail.py`

- [ ] **Step 1: Write failing pure tests for adaptive sizing and state defaults**

Append these tests to `Tests/UI/test_console_workspace_context_rail.py`:

```python
from tldw_chatbook.Workspaces.display_state import (
    CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
    ConsoleWorkspaceConversationSectionState,
    console_workspace_conversation_result_copy,
    console_workspace_conversation_visible_rows,
)


def test_console_workspace_conversation_section_state_defaults() -> None:
    section = ConsoleWorkspaceConversationSectionState(
        workspace_id="ws-a",
        collapsed=False,
        query="",
        selected_summary="No active conversation.",
        rows=(),
    )

    assert section.workspace_id == "ws-a"
    assert section.workspace_total_count is None
    assert section.result_total_count is None
    assert section.result_limit == CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT
    assert section.search_enabled is True
    assert section.new_conversation_enabled is True
    assert section.error_copy == ""


def test_console_workspace_conversation_visible_rows_are_clamped() -> None:
    assert console_workspace_conversation_visible_rows(None) == 4
    assert console_workspace_conversation_visible_rows(10) == 4
    assert console_workspace_conversation_visible_rows(48) == 7
    assert console_workspace_conversation_visible_rows(120) == 12


def test_console_workspace_conversation_result_copy_is_explicit() -> None:
    assert (
        console_workspace_conversation_result_copy(
            query="research",
            result_total_count=143,
            result_limit=50,
        )
        == "Showing 50 of 143 matches"
    )
    assert (
        console_workspace_conversation_result_copy(
            query="research",
            result_total_count=3,
            result_limit=50,
        )
        == "3 matches"
    )
    assert (
        console_workspace_conversation_result_copy(
            query="",
            result_total_count=None,
            result_limit=50,
        )
        == ""
    )
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_section_state_defaults Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_visible_rows_are_clamped Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_result_copy_is_explicit --tb=short
```

Expected: FAIL with import errors for `ConsoleWorkspaceConversationSectionState` and helper names.

- [ ] **Step 3: Add display-state constants, dataclass, and helpers**

In `tldw_chatbook/Workspaces/display_state.py`, add these definitions after `ConsoleWorkspaceConversationRow`:

```python
CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT = 50
CONSOLE_WORKSPACE_CONVERSATION_MIN_VISIBLE_ROWS = 4
CONSOLE_WORKSPACE_CONVERSATION_MAX_VISIBLE_ROWS = 12
CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT = 3
CONSOLE_WORKSPACE_CONVERSATION_HEIGHT_RATIO = 0.45


@dataclass(frozen=True)
class ConsoleWorkspaceConversationSectionState:
    """Renderable state for the Console workspace Conversations subsection."""

    workspace_id: str
    collapsed: bool
    query: str
    selected_summary: str
    rows: tuple[ConsoleWorkspaceConversationRow, ...]
    workspace_total_count: int | None = None
    result_total_count: int | None = None
    result_limit: int = CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT
    status_copy: str = ""
    empty_copy: str = ""
    search_enabled: bool = True
    new_conversation_enabled: bool = True
    error_copy: str = ""


def console_workspace_conversation_visible_rows(body_height: int | None) -> int:
    """Return the adaptive visible row count for the bounded conversation list."""

    if body_height is None or body_height <= 0:
        return CONSOLE_WORKSPACE_CONVERSATION_MIN_VISIBLE_ROWS
    target_rows = int(
        (int(body_height) * CONSOLE_WORKSPACE_CONVERSATION_HEIGHT_RATIO)
        // CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT
    )
    return max(
        CONSOLE_WORKSPACE_CONVERSATION_MIN_VISIBLE_ROWS,
        min(CONSOLE_WORKSPACE_CONVERSATION_MAX_VISIBLE_ROWS, target_rows),
    )


def console_workspace_conversation_result_copy(
    *,
    query: str,
    result_total_count: int | None,
    result_limit: int = CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
) -> str:
    """Return explicit search result count copy for the conversation rail."""

    if not str(query or "").strip() or result_total_count is None:
        return ""
    total = max(0, int(result_total_count))
    limit = max(1, int(result_limit))
    if total > limit:
        return f"Showing {limit} of {total} matches"
    return f"{total} match" if total == 1 else f"{total} matches"
```

Then extend `ConsoleWorkspaceContextState` with this field after `conversation_empty_copy`:

```python
    conversation_section: ConsoleWorkspaceConversationSectionState | None = None
```

- [ ] **Step 4: Update degraded state constructors**

In every `ConsoleWorkspaceContextState(...)` constructor inside `build_console_workspace_state`, add:

```python
            conversation_section=None,
```

Expected: all constructors still type-check at runtime and old callers can pass no section.

- [ ] **Step 5: Run tests for Task 1**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_section_state_defaults Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_visible_rows_are_clamped Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_result_copy_is_explicit --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add tldw_chatbook/Workspaces/display_state.py Tests/UI/test_console_workspace_context_rail.py
git commit -m "Add Console conversation section display state"
```

Expected: commit succeeds.

---

### Task 2: Render The Bounded And Collapsible Conversations Subsection

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Modify: `tldw_chatbook/Workspaces/display_state.py`
- Test: `Tests/UI/test_console_workspace_context_rail.py`

- [ ] **Step 1: Write failing mounted tests for expanded and collapsed rendering**

Append these tests to `Tests/UI/test_console_workspace_context_rail.py`:

```python
from dataclasses import replace

from tldw_chatbook.Workspaces.display_state import (
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationRow,
    ConsoleWorkspaceConversationSectionState,
)


def _section_state(*, collapsed: bool = False, rows: int = 6) -> ConsoleWorkspaceConversationSectionState:
    conversation_rows = tuple(
        ConsoleWorkspaceConversationRow(
            conversation_id=f"conv-{index}",
            title=f"Conversation {index}",
            status="workspace-thread",
            selected=index == 2,
        )
        for index in range(rows)
    )
    return ConsoleWorkspaceConversationSectionState(
        workspace_id="ws-a",
        collapsed=collapsed,
        query="",
        selected_summary="Conversation 2 - saved workspace",
        rows=conversation_rows,
        workspace_total_count=rows,
        result_total_count=None,
        status_copy="",
        empty_copy="No active workspace conversations.",
    )


def _base_workspace_state(section: ConsoleWorkspaceConversationSectionState) -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label="Workspace: Test",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Runtime: none",
        conversation_rows=section.rows,
        conversation_section=section,
        conversation_empty_copy="No active workspace conversations.",
        change_workspace_enabled=True,
        change_workspace_recovery="",
        new_conversation_enabled=True,
        new_conversation_recovery="",
        recovery_copy="",
    )


@pytest.mark.asyncio
async def test_console_workspace_conversations_render_bounded_expanded_section() -> None:
    app = _build_test_app()
    section = _section_state(collapsed=False, rows=8)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_workspace_state(section))
        await pilot.pause()

        assert _static_plain(console, "#console-workspace-conversations-title") == "Conversations (8)"
        assert _static_plain(console, "#console-workspace-selected-conversation") == "Conversation 2 - saved workspace"
        assert len(console.query("#console-workspace-conversation-search")) == 1
        assert len(console.query("#console-workspace-conversation-search-clear")) == 1
        assert len(console.query("#console-new-workspace-conversation")) == 1
        conversation_list = console.query_one("#console-workspace-conversations")
        rows = list(console.query(".console-workspace-conversation-row"))
        assert len(rows) == 8
        assert conversation_list.region.height <= 36


@pytest.mark.asyncio
async def test_console_workspace_conversations_collapsed_shows_selected_summary_only() -> None:
    app = _build_test_app()
    section = _section_state(collapsed=True, rows=8)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
        tray.sync_state(_base_workspace_state(section))
        await pilot.pause()

        assert _static_plain(console, "#console-workspace-conversations-title") == "Conversations (8)"
        assert _static_plain(console, "#console-workspace-selected-conversation") == "Conversation 2 - saved workspace"
        assert len(console.query("#console-workspace-conversation-search")) == 0
        assert len(console.query("#console-workspace-conversations")) == 0
        assert len(console.query("#console-new-workspace-conversation")) == 0
```

- [ ] **Step 2: Run the new rendering tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversations_render_bounded_expanded_section Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversations_collapsed_shows_selected_summary_only --tb=short
```

Expected: FAIL because the widget does not render the new header/search/collapse structure.

- [ ] **Step 3: Import the new section state and Textual widgets**

In `tldw_chatbook/Widgets/Console/console_workspace_context.py`, update imports:

```python
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from tldw_chatbook.Workspaces.display_state import (
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationSectionState,
    console_workspace_conversation_visible_rows,
)
```

- [ ] **Step 4: Add fallback section builder**

Inside `ConsoleWorkspaceContextTray`, add:

```python
    def _conversation_section(self) -> ConsoleWorkspaceConversationSectionState:
        section = self.state.conversation_section
        if section is not None:
            return section
        selected = next(
            (row for row in self.state.conversation_rows if row.selected),
            None,
        )
        selected_summary = (
            f"{self._conversation_title(selected.title)} - {self._conversation_detail_status(selected.status) or 'conversation'}"
            if selected is not None
            else "No active conversation."
        )
        return ConsoleWorkspaceConversationSectionState(
            workspace_id="",
            collapsed=False,
            query="",
            selected_summary=selected_summary,
            rows=self.state.conversation_rows,
            workspace_total_count=len(self.state.conversation_rows),
            result_total_count=None,
            status_copy="",
            empty_copy=self.state.conversation_empty_copy,
            search_enabled=True,
            new_conversation_enabled=self.state.new_conversation_enabled,
        )

    @staticmethod
    def _conversation_count_title(section: ConsoleWorkspaceConversationSectionState) -> str:
        count = section.workspace_total_count
        if count is None:
            count = len(section.rows)
        return f"Conversations ({count})"
```

- [ ] **Step 5: Replace the inline conversation rendering block**

In `compose()`, replace the current `Conversations` title, inline `Vertical(id="console-workspace-conversations")`, and immediately following `New conversation` block with:

```python
        section = self._conversation_section()
        with Horizontal(id="console-workspace-conversations-header", classes="console-workspace-conversations-header"):
            title = self._static(
                self._conversation_count_title(section),
                id="console-workspace-conversations-title",
                classes="destination-section",
            )
            title.styles.width = "1fr"
            yield title
            toggle_label = "+" if section.collapsed else "-"
            toggle = Button(
                toggle_label,
                id="console-workspace-conversations-toggle",
                classes="console-workspace-action console-workspace-conversations-toggle",
                compact=True,
            )
            toggle.tooltip = "Expand Conversations" if section.collapsed else "Collapse Conversations"
            toggle.styles.width = 3
            toggle.styles.min_width = 3
            yield toggle
        yield self._static(
            section.selected_summary or "No active conversation.",
            id="console-workspace-selected-conversation",
            classes="console-workspace-selected-conversation",
        )
        if not section.collapsed:
            with Horizontal(id="console-workspace-conversation-search-row", classes="console-workspace-conversation-search-row"):
                search_input = Input(
                    value=section.query,
                    placeholder="Search workspace conversations",
                    id="console-workspace-conversation-search",
                    classes="console-workspace-conversation-search",
                    disabled=not section.search_enabled,
                )
                search_input.styles.width = "1fr"
                yield search_input
                clear_button = Button(
                    "Clear",
                    id="console-workspace-conversation-search-clear",
                    classes="console-workspace-action console-workspace-conversation-search-clear",
                    compact=True,
                    disabled=not bool(str(section.query or "").strip()),
                )
                clear_button.tooltip = "Clear conversation search"
                yield clear_button
            if section.status_copy:
                yield self._static(
                    section.status_copy,
                    id="console-workspace-conversation-search-status",
                    classes="console-workspace-empty-copy",
                )
            if section.error_copy:
                yield self._static(
                    section.error_copy,
                    id="console-workspace-conversation-search-error",
                    classes="console-workspace-recovery",
                )
            visible_rows = console_workspace_conversation_visible_rows(
                getattr(getattr(self, "parent", None), "region", None).height
                if getattr(getattr(self, "parent", None), "region", None) is not None
                else None
            )
            conversation_list = VerticalScroll(id="console-workspace-conversations")
            conversation_list.styles.height = max(1, visible_rows * 3)
            conversation_list.styles.min_height = max(1, visible_rows * 3)
            with conversation_list:
                if section.rows:
                    for index, row in enumerate(section.rows):
                        marker = "> " if row.selected else "  "
                        title = self._conversation_title(row.title)
                        visible_title = self._conversation_visible_title(title)
                        status = self._conversation_status(row.status)
                        detail = self._conversation_detail_status(row.status)
                        status_suffix = f" [{status}]" if status else ""
                        secondary = detail or "conversation"
                        yield self._conversation_button(
                            f"{marker}{visible_title}\n  {secondary}",
                            id=f"console-workspace-conversation-{index}",
                            conversation_id=row.conversation_id,
                            tooltip_label=f"{title}{status_suffix}",
                            selected=row.selected,
                        )
                else:
                    yield self._static(
                        section.empty_copy or self.state.conversation_empty_copy,
                        id="console-workspace-empty-conversations",
                        classes="console-workspace-empty-copy",
                    )
            if section.new_conversation_enabled:
                yield Button(
                    "New conversation",
                    id="console-new-workspace-conversation",
                    classes="console-workspace-action",
                    compact=True,
                )
                if self.state.new_conversation_recovery:
                    yield self._static(
                        self.state.new_conversation_recovery,
                        id="console-new-workspace-conversation-recovery",
                        classes="console-workspace-recovery",
                    )
```

- [ ] **Step 6: Run rendering tests for Task 2**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversations_render_bounded_expanded_section Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversations_collapsed_shows_selected_summary_only --tb=short
```

Expected: PASS.

- [ ] **Step 7: Run existing workspace rail tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py --tb=short
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add tldw_chatbook/Widgets/Console/console_workspace_context.py tldw_chatbook/Workspaces/display_state.py Tests/UI/test_console_workspace_context_rail.py
git commit -m "Render bounded Console workspace conversations"
```

Expected: commit succeeds.

---

### Task 3: Add ChatScreen Conversation Section State And Collapse Preferences

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_workspace_context_rail.py`

- [ ] **Step 1: Write failing mounted tests for overflow reachability and collapse persistence**

Append these tests to `Tests/UI/test_console_workspace_context_rail.py`:

```python
@pytest.mark.asyncio
async def test_console_workspace_many_conversations_keep_lower_status_reachable() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    for index in range(40):
        service.link_membership(
            active_workspace.workspace_id,
            item_type="conversation",
            item_id=f"overflow-chat-{index}",
            role="workspace-thread",
            title=f"Overflow Chat {index:02d}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 34)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversations")

        conversation_list = console.query_one("#console-workspace-conversations")
        server_readiness = console.query_one("#console-workspace-server-readiness-label")
        assert conversation_list.region.height <= 36
        assert server_readiness.region.y > conversation_list.region.y
        assert server_readiness.region.y < console.query_one("#console-left-rail").region.y + console.query_one("#console-left-rail").region.height + 80


@pytest.mark.asyncio
async def test_console_workspace_conversation_collapse_persists_per_workspace() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    default_workspace = service.get_active_workspace()
    service.create_workspace(workspace_id="ws-collapse-b", name="Collapse B")
    service.link_membership(
        default_workspace.workspace_id,
        item_type="conversation",
        item_id="collapse-chat-a",
        role="workspace-thread",
        title="Collapse Chat A",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversations-toggle")

        await pilot.click("#console-workspace-conversations-toggle")
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversations")) == 0
        assert "Collapse Chat A" in _visible_text(console)

        service.set_active_workspace("ws-collapse-b")
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversations")) == 1

        service.set_active_workspace(default_workspace.workspace_id)
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversations")) == 0
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_many_conversations_keep_lower_status_reachable Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_collapse_persists_per_workspace --tb=short
```

Expected: FAIL because `ChatScreen` does not build section state or collapse preferences.

- [ ] **Step 3: Import section state helpers**

In `tldw_chatbook/UI/Screens/chat_screen.py`, extend the `Workspaces.display_state` import:

```python
    CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
    ConsoleWorkspaceConversationSectionState,
    console_workspace_conversation_result_copy,
```

- [ ] **Step 4: Initialize screen-owned state**

In `ChatScreen.__init__`, near existing Console fields, add:

```python
        self._console_workspace_conversation_query = ""
        self._console_workspace_conversation_search_timer: Any | None = None
        self._console_workspace_conversation_search_token = 0
        self._console_workspace_conversation_workspace_id: str | None = None
```

- [ ] **Step 5: Add conversation section preference helpers**

Add these methods near `_console_rail_state_config()`:

```python
    def _console_conversation_section_config(self) -> dict[str, Any]:
        """Return mutable Console conversation-section UI preferences."""
        console_config = self._console_config()
        section_config = console_config.get("conversation_section")
        if not isinstance(section_config, dict):
            section_config = {}
            console_config["conversation_section"] = section_config
        return section_config

    def _console_workspace_conversations_collapsed(self, workspace_id: str | None) -> bool:
        """Return stored collapse preference for one workspace."""
        key = str(workspace_id or "global").strip() or "global"
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return False
        console_config = app_config.get("console")
        if not isinstance(console_config, dict):
            return False
        section_config = console_config.get("conversation_section")
        if not isinstance(section_config, dict):
            return False
        value = section_config.get(key)
        return bool(value.get("collapsed")) if isinstance(value, dict) else False

    def _set_console_workspace_conversations_collapsed(
        self,
        workspace_id: str | None,
        collapsed: bool,
    ) -> None:
        """Store collapse preference for one workspace in memory."""
        key = str(workspace_id or "global").strip() or "global"
        section_config = self._console_conversation_section_config()
        section_config[key] = {"collapsed": bool(collapsed)}
```

- [ ] **Step 6: Add selected summary and row merging helpers**

Add these methods near `_with_native_console_session_rows()`:

```python
    @staticmethod
    def _console_workspace_row_key(row: ConsoleWorkspaceConversationRow) -> str:
        return str(row.conversation_id or "").strip()

    def _selected_console_workspace_conversation_summary(
        self,
        rows: list[ConsoleWorkspaceConversationRow],
    ) -> str:
        selected = next((row for row in rows if row.selected), None)
        if selected is None:
            return "No active conversation."
        title = ConsoleWorkspaceContextTray._conversation_title(selected.title)
        detail = ConsoleWorkspaceContextTray._conversation_detail_status(selected.status)
        return f"{title} - {detail or 'conversation'}"

    def _merge_console_workspace_rows(
        self,
        primary: list[ConsoleWorkspaceConversationRow],
        secondary: list[ConsoleWorkspaceConversationRow],
    ) -> list[ConsoleWorkspaceConversationRow]:
        merged: list[ConsoleWorkspaceConversationRow] = []
        seen: set[str] = set()
        for row in primary + secondary:
            key = self._console_workspace_row_key(row)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(row)
        return merged
```

- [ ] **Step 7: Build conversation section state**

Add this method near `_build_console_workspace_context_state()`:

```python
    def _with_console_workspace_conversation_section(
        self,
        state: ConsoleWorkspaceContextState,
    ) -> ConsoleWorkspaceContextState:
        """Attach renderable Conversations subsection state to workspace context."""
        workspace_id = ""
        store = self._console_chat_store
        if store is not None and store.workspace_context.active_workspace_id:
            workspace_id = str(store.workspace_context.active_workspace_id)
        elif state.workspace_label.startswith("Workspace: "):
            workspace_id = state.workspace_label.removeprefix("Workspace: ").strip()

        if self._console_workspace_conversation_workspace_id != workspace_id:
            self._console_workspace_conversation_query = ""
            self._console_workspace_conversation_workspace_id = workspace_id

        rows = list(state.conversation_rows)
        selected_summary = self._selected_console_workspace_conversation_summary(rows)
        query = self._console_workspace_conversation_query
        result_total = len(rows) if query.strip() else None
        status_copy = console_workspace_conversation_result_copy(
            query=query,
            result_total_count=result_total,
            result_limit=CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
        )
        section = ConsoleWorkspaceConversationSectionState(
            workspace_id=workspace_id,
            collapsed=self._console_workspace_conversations_collapsed(workspace_id),
            query=query,
            selected_summary=selected_summary,
            rows=tuple(rows),
            workspace_total_count=len(rows),
            result_total_count=result_total,
            status_copy=status_copy,
            empty_copy=(
                "No matches in this workspace."
                if query.strip()
                else state.conversation_empty_copy
            ),
            search_enabled=True,
            new_conversation_enabled=state.new_conversation_enabled,
        )
        return replace(state, conversation_section=section)
```

- [ ] **Step 8: Attach the section in the workspace state builder**

Change `_build_console_workspace_context_state()` from:

```python
        return self._with_native_console_session_rows(state)
```

to:

```python
        state = self._with_native_console_session_rows(state)
        return self._with_console_workspace_conversation_section(state)
```

- [ ] **Step 9: Handle collapse toggle button**

In `on_button_pressed`, before the `console-new-workspace-conversation` branch, add:

```python
        if button_id == "console-workspace-conversations-toggle":
            event.stop()
            state = self._build_console_workspace_context_state()
            section = state.conversation_section
            workspace_id = section.workspace_id if section is not None else None
            collapsed = not bool(section.collapsed if section is not None else False)
            self._set_console_workspace_conversations_collapsed(workspace_id, collapsed)
            self._sync_console_workspace_context()
            return
```

- [ ] **Step 10: Run Task 3 tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_many_conversations_keep_lower_status_reachable Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_collapse_persists_per_workspace --tb=short
```

Expected: PASS.

- [ ] **Step 11: Run existing workspace rail tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py --tb=short
```

Expected: PASS.

- [ ] **Step 12: Commit Task 3**

Run:

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_workspace_context_rail.py
git commit -m "Persist Console conversation rail collapse state"
```

Expected: commit succeeds.

---

### Task 4: Add Active Workspace Conversation Search

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing membership search scope test**

Append this test to `Tests/UI/test_console_native_chat_flow.py`:

```python
@pytest.mark.asyncio
async def test_console_workspace_conversation_search_filters_active_workspace_memberships():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    other_workspace = service.create_workspace(workspace_id="ws-other-search", name="Other Search")
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="member-alpha",
        role="workspace-thread",
        title="Alpha membership conversation",
    )
    service.link_membership(
        other_workspace.workspace_id,
        item_type="conversation",
        item_id="member-other-alpha",
        role="workspace-thread",
        title="Alpha other workspace",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        await pilot.click("#console-workspace-conversation-search")
        await pilot.press("a", "l", "p", "h", "a")
        await _wait_for_workspace_conversation_text(console, pilot, "Alpha membership", selected=False)
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Alpha membership" in text for text in row_texts)
        assert all("Alpha other workspace" not in text for text in row_texts)
        assert "matches" in _visible_text(console)
```

- [ ] **Step 2: Run the membership search scope test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_filters_active_workspace_memberships --tb=short
```

Expected: FAIL because no search handler updates the rail.

- [ ] **Step 3: Add search input handler**

In `tldw_chatbook/UI/Screens/chat_screen.py`, add a specific handler near other `Input.Changed` handlers:

```python
    @on(Input.Changed, "#console-workspace-conversation-search")
    def on_console_workspace_conversation_search_changed(self, event: Input.Changed) -> None:
        """Debounce active-workspace conversation search in the Console rail."""
        event.stop()
        self._console_workspace_conversation_query = str(event.value or "")
        self._console_workspace_conversation_search_token += 1
        token = self._console_workspace_conversation_search_token
        workspace_id = self._active_console_workspace_id_for_conversation_search()
        query = self._console_workspace_conversation_query
        if self._console_workspace_conversation_search_timer is not None:
            self._console_workspace_conversation_search_timer.stop()
        self._console_workspace_conversation_search_timer = self.set_timer(
            0.2,
            lambda: self.run_worker(
                self._refresh_console_workspace_conversation_search(
                    workspace_id,
                    query,
                    token,
                ),
                exclusive=True,
            ),
        )
```

- [ ] **Step 4: Add clear-search handler**

In `on_button_pressed`, before row selection, add:

```python
        if button_id == "console-workspace-conversation-search-clear":
            event.stop()
            self._console_workspace_conversation_query = ""
            self._console_workspace_conversation_search_token += 1
            self._sync_console_workspace_context()
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
            return
```

- [ ] **Step 5: Add active workspace and focus helpers**

Add these methods near `_current_console_workspace_context()`:

```python
    def _active_console_workspace_id_for_conversation_search(self) -> str:
        """Return the current active workspace id for Console conversation search."""
        store = self._console_chat_store
        if store is not None and store.workspace_context.active_workspace_id:
            return str(store.workspace_context.active_workspace_id)
        service = getattr(self.app_instance, "workspace_registry_service", None)
        get_active_workspace = getattr(service, "get_active_workspace", None)
        if callable(get_active_workspace):
            try:
                workspace = get_active_workspace()
            except Exception:
                logger.debug("Unable to read active workspace for conversation search", exc_info=True)
                return ""
            return str(getattr(workspace, "workspace_id", "") or "")
        return ""

    def _focus_console_workspace_conversation_search(self) -> None:
        """Restore focus to the conversation search input when it is mounted."""
        try:
            search = self.query_one("#console-workspace-conversation-search", Input)
        except (NoMatches, QueryError):
            return
        search.focus()
```

- [ ] **Step 6: Add search row builders**

Add these methods near `_merge_console_workspace_rows()`:

```python
    def _native_console_rows_for_workspace_search(
        self,
        workspace_id: str,
        query: str,
    ) -> list[ConsoleWorkspaceConversationRow]:
        """Return matching open native sessions for the active workspace search."""
        store = self._console_chat_store
        if store is None:
            return []
        needle = str(query or "").strip().lower()
        rows: list[ConsoleWorkspaceConversationRow] = []
        for session in store.sessions():
            if workspace_id and str(session.workspace_id or "") != workspace_id:
                continue
            title = str(session.title or "Untitled conversation")
            if needle and needle not in title.lower():
                continue
            conversation_id = (
                str(session.persisted_conversation_id)
                if session.persisted_conversation_id
                else f"native:{session.id}"
            )
            rows.append(
                ConsoleWorkspaceConversationRow(
                    conversation_id=conversation_id,
                    title=title,
                    status="active" if session.id == store.active_session_id else "open",
                    selected=session.id == store.active_session_id,
                )
            )
        return rows

    def _membership_console_rows_for_workspace_search(
        self,
        workspace_id: str,
        query: str,
    ) -> list[ConsoleWorkspaceConversationRow]:
        """Return matching workspace conversation membership rows."""
        service = getattr(self.app_instance, "workspace_registry_service", None)
        list_conversations = getattr(service, "list_workspace_conversations", None)
        if not callable(list_conversations) or not workspace_id:
            return []
        needle = str(query or "").strip().lower()
        try:
            memberships = list_conversations(workspace_id)
        except Exception:
            logger.debug("Unable to search workspace conversation memberships", exc_info=True)
            return []
        rows: list[ConsoleWorkspaceConversationRow] = []
        current_conversation = self._current_console_conversation_id()
        for membership in memberships:
            title = str(getattr(membership, "title", "") or getattr(membership, "item_id", ""))
            if needle and needle not in title.lower():
                continue
            conversation_id = str(getattr(membership, "item_id", "") or "")
            rows.append(
                ConsoleWorkspaceConversationRow(
                    conversation_id=conversation_id,
                    title=title,
                    status=str(getattr(membership, "role", "") or "workspace-thread"),
                    selected=bool(current_conversation and conversation_id == current_conversation),
                )
            )
        return rows
```

- [ ] **Step 7: Add guarded refresh**

Add this async method near the search row builders:

```python
    async def _refresh_console_workspace_conversation_search(
        self,
        workspace_id: str,
        query: str,
        token: int,
    ) -> None:
        """Refresh search results only if workspace and query are still current."""
        if token != self._console_workspace_conversation_search_token:
            return
        if workspace_id != self._active_console_workspace_id_for_conversation_search():
            return
        if query != self._console_workspace_conversation_query:
            return
        self._sync_console_workspace_context()
        self.call_after_refresh(self._focus_console_workspace_conversation_search)
```

This refresh method applies the stale-result guard and triggers a sync. Persisted service query merging is added in Task 5.

- [ ] **Step 8: Change section builder to use native and membership search rows while query is active**

In `_with_console_workspace_conversation_section`, before creating `section`, replace:

```python
        rows = list(state.conversation_rows)
```

with:

```python
        rows = list(state.conversation_rows)
        if self._console_workspace_conversation_query.strip():
            native_rows = self._native_console_rows_for_workspace_search(
                workspace_id,
                self._console_workspace_conversation_query,
            )
            membership_rows = self._membership_console_rows_for_workspace_search(
                workspace_id,
                self._console_workspace_conversation_query,
            )
            rows = self._merge_console_workspace_rows(native_rows, membership_rows)
```

- [ ] **Step 9: Run the membership search scope test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_filters_active_workspace_memberships --tb=short
```

Expected: PASS.

- [ ] **Step 10: Commit Task 4 search foundation**

Run:

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Add Console conversation rail search controls"
```

Expected: commit succeeds with the membership/native search test passing.

---

### Task 5: Merge Persisted Search Results, Caps, Errors, And Stale Guards

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing tests for persisted result cap, empty copy, and stale guard**

Append these tests to `Tests/UI/test_console_native_chat_flow.py`:

```python
class SearchableConversationService(StaticConversationTreeService):
    def __init__(self, conversations: dict[str, dict]) -> None:
        super().__init__(conversations)
        self.list_calls: list[dict[str, object]] = []

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        query = str(kwargs.get("query") or "").strip().lower()
        workspace_id = str(kwargs.get("workspace_id") or "").strip()
        limit = int(kwargs.get("limit") or 50)
        items = []
        for conversation_id, tree in self.conversations.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            if workspace_id and str(conversation.get("workspace_id") or "") != workspace_id:
                continue
            if query and query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "total": len(items),
            "limit": limit,
            "offset": 0,
        }


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_scopes_persisted_results_to_active_workspace():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    other_workspace = service.create_workspace(workspace_id="ws-other-search", name="Other Search")
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "persisted-alpha": {
                "conversation": {
                    "id": "persisted-alpha",
                    "title": "Alpha persisted conversation",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [],
            },
            "other-alpha": {
                "conversation": {
                    "id": "other-alpha",
                    "title": "Alpha other workspace",
                    "workspace_id": other_workspace.workspace_id,
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        await pilot.click("#console-workspace-conversation-search")
        await pilot.press("a", "l", "p", "h", "a")
        await _wait_for_workspace_conversation_text(console, pilot, "Alpha persisted", selected=False)
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Alpha persisted" in text for text in row_texts)
        assert all("Alpha other workspace" not in text for text in row_texts)
        assert app.chat_conversation_scope_service.list_calls[-1]["workspace_id"] == active_workspace.workspace_id


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_shows_cap_and_empty_copy():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    conversations = {
        f"topic-{index}": {
            "conversation": {
                "id": f"topic-{index}",
                "title": f"Topic conversation {index:02d}",
                "workspace_id": active_workspace.workspace_id,
            },
            "root_threads": [],
        }
        for index in range(60)
    }
    app.chat_conversation_scope_service = SearchableConversationService(conversations)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        await pilot.click("#console-workspace-conversation-search")
        await pilot.press("t", "o", "p", "i", "c")
        await _wait_for_text(console, pilot, "Showing 50 of 60 matches")

        search = console.query_one("#console-workspace-conversation-search", Input)
        search.value = "missing"
        await pilot.pause(0.3)
        await _wait_for_text(console, pilot, "No matches in this workspace.")


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_ignores_stale_workspace_results():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.create_workspace(workspace_id="ws-stale-b", name="Stale B")
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "stale-a": {
                "conversation": {
                    "id": "stale-a",
                    "title": "Stale Alpha",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        console._console_workspace_conversation_query = "Alpha"
        stale_token = console._console_workspace_conversation_search_token + 1
        console._console_workspace_conversation_search_token = stale_token
        service.set_active_workspace("ws-stale-b")
        await console._refresh_console_workspace_conversation_search(
            active_workspace.workspace_id,
            "Alpha",
            stale_token,
        )
        assert "Stale Alpha" not in _visible_text(console)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_scopes_persisted_results_to_active_workspace Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_shows_cap_and_empty_copy Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_ignores_stale_workspace_results --tb=short
```

Expected: FAIL because persisted results, caps, and empty copy are not wired.

- [ ] **Step 3: Add persisted search cache fields**

In `ChatScreen.__init__`, add:

```python
        self._console_workspace_conversation_search_rows: tuple[ConsoleWorkspaceConversationRow, ...] = ()
        self._console_workspace_conversation_search_total: int | None = None
        self._console_workspace_conversation_search_error = ""
```

- [ ] **Step 4: Add persisted search method**

Add this method near `_membership_console_rows_for_workspace_search()`:

```python
    async def _persisted_console_rows_for_workspace_search(
        self,
        workspace_id: str,
        query: str,
    ) -> tuple[list[ConsoleWorkspaceConversationRow], int | None, str]:
        """Return persisted workspace conversation search rows, total, and error copy."""
        scope_service = getattr(self.app_instance, "chat_conversation_scope_service", None)
        list_conversations = getattr(scope_service, "list_conversations", None)
        if not callable(list_conversations) or not workspace_id:
            return [], 0, ""
        try:
            result = list_conversations(
                mode="local",
                query=query,
                scope_type="workspace",
                workspace_id=workspace_id,
                limit=CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
                offset=0,
            )
            result = await result if inspect.isawaitable(result) else result
        except Exception:
            logger.exception("Unable to search Console workspace conversations")
            return [], None, "Workspace conversation search is unavailable."
        if not isinstance(result, dict):
            return [], 0, ""
        items = result.get("items")
        if not isinstance(items, list):
            items = []
        total = result.get("total")
        try:
            total_count = int(total)
        except (TypeError, ValueError):
            total_count = len(items)
        current_conversation = self._current_console_conversation_id()
        rows: list[ConsoleWorkspaceConversationRow] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            conversation_id = str(item.get("id") or "").strip()
            if not conversation_id:
                continue
            rows.append(
                ConsoleWorkspaceConversationRow(
                    conversation_id=conversation_id,
                    title=str(item.get("title") or "Untitled conversation"),
                    status=str(item.get("state") or "workspace-thread"),
                    selected=bool(current_conversation and current_conversation == conversation_id),
                )
            )
        return rows, total_count, ""
```

- [ ] **Step 5: Complete `_refresh_console_workspace_conversation_search()`**

Replace the body of `_refresh_console_workspace_conversation_search()` after the stale guards with:

```python
        native_rows = self._native_console_rows_for_workspace_search(workspace_id, query)
        membership_rows = self._membership_console_rows_for_workspace_search(workspace_id, query)
        persisted_rows, persisted_total, error_copy = await self._persisted_console_rows_for_workspace_search(
            workspace_id,
            query,
        )
        if token != self._console_workspace_conversation_search_token:
            return
        if workspace_id != self._active_console_workspace_id_for_conversation_search():
            return
        if query != self._console_workspace_conversation_query:
            return
        merged = self._merge_console_workspace_rows(
            self._merge_console_workspace_rows(native_rows, membership_rows),
            persisted_rows,
        )
        self._console_workspace_conversation_search_rows = tuple(merged)
        self._console_workspace_conversation_search_total = persisted_total
        self._console_workspace_conversation_search_error = error_copy
        self._sync_console_workspace_context()
        self.call_after_refresh(self._focus_console_workspace_conversation_search)
```

- [ ] **Step 6: Update section builder to use persisted cache**

In `_with_console_workspace_conversation_section`, replace the query-active row block from Task 4 with:

```python
        if self._console_workspace_conversation_query.strip():
            rows = list(self._console_workspace_conversation_search_rows)
```

Then replace the `result_total` and `status_copy` calculation with:

```python
        query = self._console_workspace_conversation_query
        result_total = self._console_workspace_conversation_search_total if query.strip() else None
        if query.strip() and result_total is None and not self._console_workspace_conversation_search_error:
            result_total = len(rows)
        status_copy = console_workspace_conversation_result_copy(
            query=query,
            result_total_count=result_total,
            result_limit=CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
        )
```

Set `error_copy` in the dataclass constructor:

```python
            error_copy=self._console_workspace_conversation_search_error,
```

- [ ] **Step 7: Reset search cache when query clears or workspace changes**

In `_with_console_workspace_conversation_section`, inside the workspace-change branch, add:

```python
            self._console_workspace_conversation_search_rows = ()
            self._console_workspace_conversation_search_total = None
            self._console_workspace_conversation_search_error = ""
```

In the clear-search button branch, add the same three assignments before `_sync_console_workspace_context()`.

- [ ] **Step 8: Run Task 5 tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_scopes_persisted_results_to_active_workspace Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_shows_cap_and_empty_copy Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_ignores_stale_workspace_results --tb=short
```

Expected: PASS.

- [ ] **Step 9: Commit Task 5**

Run:

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Search Console workspace conversations"
```

Expected: commit succeeds.

---

### Task 6: Preserve Search During Selection And Reset On Workspace Switch

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing selection and workspace switch tests**

Append these tests to `Tests/UI/test_console_native_chat_flow.py`:

```python
@pytest.mark.asyncio
async def test_console_workspace_conversation_search_selection_keeps_query_active():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "select-alpha": {
                "conversation": {
                    "id": "select-alpha",
                    "title": "Select Alpha",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "select-alpha-message",
                        "conversation_id": "select-alpha",
                        "role": "user",
                        "sender": "user",
                        "content": "selected alpha prompt",
                        "children": [],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await pilot.click("#console-workspace-conversation-search")
        await pilot.press("a", "l", "p", "h", "a")
        await _wait_for_workspace_conversation_text(console, pilot, "Select Alpha", selected=False)

        await _click_console_workspace_conversation_for_id(console, pilot, "select-alpha")

        await _wait_for_text(console, pilot, "selected alpha prompt")
        search = console.query_one("#console-workspace-conversation-search", Input)
        assert search.value == "alpha"
        assert "Select Alpha" in _static_plain_text(console.query_one("#console-workspace-selected-conversation", Static))


@pytest.mark.asyncio
async def test_console_workspace_switch_clears_conversation_search_and_restores_collapse_preference():
    app = _build_test_app()
    service = app.workspace_registry_service
    workspace_a = service.get_active_workspace()
    workspace_b = service.create_workspace(workspace_id="ws-search-reset", name="Search Reset")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await pilot.click("#console-workspace-conversation-search")
        await pilot.press("a", "l", "p", "h", "a")
        await pilot.click("#console-workspace-conversations-toggle")
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversation-search")) == 0

        service.set_active_workspace(workspace_b.workspace_id)
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversation-search")) == 1
        assert console.query_one("#console-workspace-conversation-search", Input).value == ""

        service.set_active_workspace(workspace_a.workspace_id)
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversation-search")) == 0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_selection_keeps_query_active Tests/UI/test_console_native_chat_flow.py::test_console_workspace_switch_clears_conversation_search_and_restores_collapse_preference --tb=short
```

Expected: FAIL if query restoration, selected summary, or workspace reset is incomplete.

- [ ] **Step 3: Preserve query during row selection**

In the `console-workspace-conversation-` branch of `on_button_pressed`, after every `await self._sync_native_console_chat_ui()` call, add:

```python
                self.call_after_refresh(self._focus_console_workspace_conversation_search)
```

Also add the same call after the branch that switches an already-open native session:

```python
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
```

- [ ] **Step 4: Reset search token on workspace change**

In `_with_console_workspace_conversation_section`, inside the workspace-change branch, add:

```python
            self._console_workspace_conversation_search_token += 1
```

Expected: any pending search result for the previous workspace becomes stale.

- [ ] **Step 5: Run Task 6 tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_search_selection_keeps_query_active Tests/UI/test_console_native_chat_flow.py::test_console_workspace_switch_clears_conversation_search_and_restores_collapse_preference --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Preserve Console conversation search state"
```

Expected: commit succeeds.

---

### Task 7: Add Console Conversation Subsection Styling

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_workspace_context_rail.py`

- [ ] **Step 1: Write failing source-level style test**

Append this test to `Tests/UI/test_console_workspace_context_rail.py`:

```python
def test_console_workspace_conversation_subsection_styles_are_declared() -> None:
    css = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text()

    assert "#console-workspace-conversations-header" in css
    assert "#console-workspace-selected-conversation" in css
    assert "#console-workspace-conversation-search-row" in css
    assert "#console-workspace-conversations" in css
    assert "scrollbar-size: 1 1" in css
```

Also add `from pathlib import Path` near the top of the test file if it is not present.

- [ ] **Step 2: Run the style test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_subsection_styles_are_declared --tb=short
```

Expected: FAIL because the selectors are not declared.

- [ ] **Step 3: Add component TCSS rules**

In `tldw_chatbook/css/components/_agentic_terminal.tcss`, near existing `#console-workspace-conversations` rules, add:

```css
#console-workspace-conversations-header {
    width: 100%;
    min-width: 0;
    height: 1;
    min-height: 1;
    margin: 1 0 0 0;
    layout: horizontal;
    align: left middle;
}

.console-workspace-conversations-toggle {
    width: 3;
    min-width: 3;
    max-width: 3;
    margin: 0;
}

#console-workspace-selected-conversation {
    height: auto;
    min-height: 1;
    margin: 0 0 1 0;
    padding: 0 1;
    color: $ds-text-muted;
    background: $ds-surface-panel;
}

#console-workspace-conversation-search-row {
    width: 100%;
    min-width: 0;
    height: 1;
    min-height: 1;
    margin: 0 0 1 0;
    layout: horizontal;
    align: left middle;
}

#console-workspace-conversation-search {
    width: 1fr;
    min-width: 0;
    height: 1;
    min-height: 1;
    margin: 0 1 0 0;
}

#console-workspace-conversation-search-clear {
    width: auto;
    min-width: 7;
    height: 1;
    min-height: 1;
    margin: 0;
}

#console-workspace-conversations {
    width: 100%;
    min-width: 0;
    min-height: 4;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-size: 1 1;
    scrollbar-background: $ds-surface-panel;
    scrollbar-color: $ds-grid-line;
    scrollbar-color-hover: $ds-grid-line;
    scrollbar-color-active: $ds-action-focus;
}
```

- [ ] **Step 4: Mirror or regenerate modular TCSS**

Run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` updates with the same selectors.

If the build script fails because the local environment lacks a dependency, copy the same selector block into `tldw_chatbook/css/tldw_cli_modular.tcss` under the existing Console section and record the build failure in the task notes.

- [ ] **Step 5: Run the style test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_conversation_subsection_styles_are_declared --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

Run:

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_workspace_context_rail.py
git commit -m "Style Console conversation rail subsection"
```

Expected: commit succeeds.

---

### Task 8: Run Focused Regression Suite And Capture Evidence

**Files:**
- Modify: `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-evidence.md`
- Add screenshots under: `Docs/superpowers/qa/console-uat-parallelization/`
- Modify: `backlog/tasks/task-138 - Fix-Console-conversation-rail-overflow.md`

- [ ] **Step 1: Run focused Console workspace tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py -k "workspace_conversation or workspace_rail or workspace_switch or default_workspace or conversation_search" --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run related rail/layout tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_session_settings.py::test_console_left_rail_body_scrolls_below_fixed_header_without_settings_summary Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space --tb=short
```

Expected: PASS.

- [ ] **Step 3: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Capture Textual-web/CDP evidence**

Use the repo’s existing Textual-web/CDP runbook from prior Console QA. Capture at least one screenshot with:

- 30 or more active-workspace conversations.
- Expanded Conversations section visibly bounded.
- Lower workspace status/server/handoff content reachable in the left rail.
- Search query active with result count copy visible.

Save screenshots under:

```text
Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-cdp-2026-06-25.png
Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-search-cdp-2026-06-25.png
```

- [ ] **Step 5: Write evidence note**

Create `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-evidence.md`:

```markdown
# TASK-138 Console Conversation Rail Overflow Evidence

Date: 2026-06-25

## Verification

- Focused Console workspace tests passed:
  `.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py -k "workspace_conversation or workspace_rail or workspace_switch or default_workspace or conversation_search" --tb=short`
- Rail/layout tests passed:
  `.venv/bin/python -m pytest -q Tests/UI/test_console_session_settings.py::test_console_left_rail_body_scrolls_below_fixed_header_without_settings_summary Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space --tb=short`
- `git diff --check` passed.

## Rendered Evidence

- `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-cdp-2026-06-25.png`
- `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-search-cdp-2026-06-25.png`

## ADR Check

ADR required: no
ADR path: N/A
Reason: presentation and UI preference state only; no schema, sync, workspace ownership, provider/runtime, or handoff contract change.
```

- [ ] **Step 6: Update TASK-138 acceptance criteria and implementation notes**

Edit `backlog/tasks/task-138 - Fix-Console-conversation-rail-overflow.md`:

- Change all AC checkboxes from `- [ ]` to `- [x]`.
- Add implementation notes:

```markdown
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a bounded, adaptive Conversations subsection to the Console workspace rail so large active-workspace conversation sets cannot hide lower workspace status, server readiness, or handoff content.
- Added per-workspace collapse preference with collapsed selected-conversation summary, while keeping workspace-scoped `New conversation` available in expanded mode.
- Added transient active-workspace conversation search across open native sessions, active workspace memberships, and persisted workspace-scoped conversations, with result caps, empty/error copy, and stale workspace/query guards.
- Preserved existing workspace-scoped resume/new-chat behavior, Default workspace policy, and local-first server/sync/handoff copy.
- Added mounted regressions and Textual-web/CDP evidence for overflow, collapse, search scope, row selection, and workspace switching.
- ADR check completed: no ADR required because this is presentation and UI preference state only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
```

- [ ] **Step 7: Mark TASK-138 Done**

Run:

```bash
backlog task edit 134 -s Done
```

Expected: `TASK-138` status becomes `Done`.

- [ ] **Step 8: Commit final verification artifacts and task update**

Run:

```bash
git add Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-evidence.md Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-cdp-2026-06-25.png Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-search-cdp-2026-06-25.png "backlog/tasks/task-138 - Fix-Console-conversation-rail-overflow.md"
git commit -m "Verify Console conversation rail overflow fix"
```

Expected: commit succeeds.

---

## Final Verification

Run before claiming completion:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py -k "workspace_conversation or workspace_rail or workspace_switch or default_workspace or conversation_search" --tb=short
.venv/bin/python -m pytest -q Tests/UI/test_console_session_settings.py::test_console_left_rail_body_scrolls_below_fixed_header_without_settings_summary Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space --tb=short
git diff --check
```

Expected:

- Focused Console workspace/search tests pass.
- Related rail/layout tests pass.
- `git diff --check` exits 0.
- Textual-web/CDP evidence files exist and show the bounded conversation list with lower rail content reachable.

## Plan Self-Review

Spec coverage:

- Adaptive bounded list: Tasks 1, 2, 3, 7, 8.
- Per-workspace collapse and selected summary: Tasks 2, 3, 6.
- Active-workspace search across native sessions, memberships, and persisted conversations: Tasks 4, 5.
- No cross-workspace leakage and stale-result guard: Tasks 4, 5, 6.
- Selection keeps search active: Task 6.
- Expanded `New conversation`, collapsed hide behavior: Task 2.
- Explicit cap, empty, and error copy: Task 5.
- Mounted tests and rendered evidence: Task 8.
- ADR and Backlog hygiene: Task 0 and Task 8.

Incomplete-marker scan:

- No incomplete sections or undefined task references.

Type consistency:

- `ConsoleWorkspaceConversationSectionState`, helper names, selector IDs, and `ChatScreen` field names are consistent across tasks.
