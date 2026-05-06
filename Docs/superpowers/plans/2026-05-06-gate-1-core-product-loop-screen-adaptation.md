# Gate 1 Core Product Loop Screen Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the `Home`, `Console`, and `Library` screens into a usable core product loop that follows the approved destination layout contracts.

**Architecture:** Keep existing route IDs, service adapters, and legacy screen behavior intact while adding contract-aligned wrapper structure and mounted UI regressions. `Home` becomes a dashboard/control layout, `Console` becomes the live-work shell around existing chat functionality, and `Library` keeps its verified three-region layout while gaining actionable local modes.

**Tech Stack:** Python 3.12, Textual, pytest, Backlog.md, existing `tldw_chatbook` screen wrappers and service adapters.

---

## Scope

This plan implements Gate 1 from `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`.

Included:

- `Home`: dashboard regions, selected active-work inspector, status/authority row, and compatibility with existing controls.
- `Console`: destination header, status/mode rows, staged-context tray, transcript region, live-work inspector, and existing chat surface preservation.
- `Library`: actionable mode selection for the already-verified Library contract shell.
- QA evidence and Backlog tracking for Gate 1.

Excluded:

- Full rewrite of `ChatWindowEnhanced`.
- Full Search/RAG implementation inside Library.
- Rebuilding Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, or Settings. Those belong to Gate 2 and Gate 3 follow-up plans.

Required deferred follow-up gates:

- **Gate 1.5: Console internals decomposition and `ChatWindowEnhanced` replacement.** Gate 1 may frame the existing chat surface inside the Console shell for compatibility, but the legacy `ChatWindowEnhanced` implementation must be decomposed and replaced in a later required gate. That gate owns visual fit, keyboard flow, provider/model controls, transcript rendering, composer behavior, staged context, RAG controls, tool calls, approvals, artifacts/Chatbook save controls, and parity with existing chat features.
- **Gate 1.6: Library-native Search/RAG workflow.** Gate 1 may make Library's Search/RAG mode selectable, but the full Search/RAG implementation must be delivered in a later required gate. That gate owns source selection, RAG query input, retrieval status, evidence/results list, citations/provenance, failure/setup recovery, and handoff into Console with staged evidence.
- These gates must receive their own implementation plans and Backlog tasks before broad Gate 2 destination rewrites, because Console usability and Library retrieval are core-loop dependencies.

## File Structure

### Create

- `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
  - Mounted UI regressions for Home, Console, and Library Gate 1 contract behavior.
- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md`
  - QA walkthrough evidence for the adapted core loop.
- `backlog/tasks/task-10.4 - Product-Maturity-Phase-3.4-Core-Product-Loop-Screen-Adaptation.md`
  - Backlog task for this implementation slice, if Backlog.md does not assign a different child id.

### Modify

- `tldw_chatbook/UI/Screens/home_screen.py`
  - Reshape Home composition while preserving `HOME_CONTROL_METHODS` behavior.
- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Add Console shell regions around `ChatWindowEnhanced`.
- `tldw_chatbook/UI/Screens/library_screen.py`
  - Convert Library mode chips into actionable mode controls and render mode-specific detail/inspector copy.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add or verify shared design-system layout hooks for the new Home, Console, and Library regions.
- `tldw_chatbook/Home/dashboard_state.py`
  - Only if needed: add small pure helpers for active item selection. Do not move UI rendering here.
- `tldw_chatbook/Chat/console_live_work.py`
  - Only if needed: add small typed display-state helpers for staged context/readiness labels. Do not add UI widgets here.
- `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`
  - Update status line or add implementation note after Gate 1 is verified.
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
  - Add Gate 1 evidence and task reference.
- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
  - Link Gate 1 evidence.
- `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
  - Add parent progress note after verification.

### Read Before Editing

- `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`
- `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- `Docs/superpowers/specs/2026-05-02-new-user-first-run-shell-ux-design.md`
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- `Tests/UI/test_home_screen.py`
- `Tests/UI/test_destination_shells.py`
- `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`
- `Tests/UI/test_chat_first_handoffs.py`
- `Tests/UI/test_master_shell_design_system_contract.py`

Use the repo-level virtualenv when working in a sibling worktree:

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

---

### Task 1: Create Gate 1 Mounted Regression Skeleton

**Files:**
- Create: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
- Read: `Tests/UI/test_home_screen.py`
- Read: `Tests/UI/test_destination_shells.py`
- Read: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`

- [ ] **Step 1: Add shared test helpers**

Create a new test file with small helpers copied or imported from existing UI tests.

```python
from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _wait_for_library_snapshot,
)
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _visible_text(screen) -> str:
    static_text = [
        _static_text(widget)
        for widget in screen.query(Static)
        if widget.display and hasattr(widget, "renderable")
    ]
    button_text = [
        str(button.label)
        for button in screen.query(Button)
        if button.display and button.label is not None
    ]
    return " ".join([*static_text, *button_text])


class ConsoleHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))
```

- [ ] **Step 2: Add failing Home contract test**

Add a mounted test that proves Home needs dashboard regions and selected-item inspector.

```python
@pytest.mark.asyncio
async def test_home_core_loop_uses_dashboard_regions_and_selected_item_inspector():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_run_count=2,
        running_run_count=1,
        failed_run_count=1,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:watchlist-run:daily",
                title="Daily papers",
                source="watchlists",
                status="running",
                detail_route="subscriptions",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="local:chatbook:summary",
                title="RAG Summary Chatbook",
                source="artifacts",
                status="ready",
                detail_route="artifacts",
                console_available=True,
            ),
        ),
    )
    host = HomeHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        for selector in (
            "#home-status-row",
            "#home-scope-filter-row",
            "#home-dashboard-grid",
            "#home-attention-queue",
            "#home-active-work-region",
            "#home-inspector",
            "#home-next-actions-region",
            "#home-recent-work-region",
        ):
            assert home.query_one(selector)

        text = _visible_text(home)
        assert "Daily papers" in text
        assert "RAG Summary Chatbook" in text
        assert "Selected item" in text
        assert "Open in Console" in text
        assert "Open Chatbook in Console" in text
```

- [ ] **Step 3: Add failing Console contract test**

Add a mounted test that proves Console needs shell regions around the existing chat surface.

```python
@pytest.mark.asyncio
async def test_console_core_loop_exposes_agentic_shell_regions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        await pilot.pause(0.45)
        console = host.screen_stack[-1]

        for selector in (
            "#console-shell",
            "#console-title",
            "#console-status-row",
            "#console-mode-bar",
            "#console-workspace-grid",
            "#console-staged-context-tray",
            "#console-transcript-region",
            "#console-run-inspector",
            "#console-composer-region",
            "#chat-window",
        ):
            assert console.query_one(selector)

        text = _visible_text(console)
        assert "Console" in text
        assert "Live work" in text
        assert "Staged Context" in text
        assert "Run Inspector" in text
```

- [ ] **Step 4: Add failing Library mode test**

Add a mounted test that clicks Search/RAG and verifies Library detail/inspector copy updates without leaving the Library shell.

```python
@pytest.mark.asyncio
async def test_library_core_loop_modes_are_actionable_without_leaving_library():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-mode-search")
        await pilot.pause(0.1)

        assert screen.query_one("#library-source-detail")
        assert screen.query_one("#library-source-inspector")
        text = _visible_text(screen)
        assert "Search/RAG mode" in text
        assert "Ask in Console" in text or "Use in Console" in text
```

- [ ] **Step 5: Run focused red tests**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py --tb=short
```

Expected before implementation: fail because the new Home, Console, and Library selectors/behaviors do not exist yet.

- [ ] **Step 6: Commit the red tests**

```bash
git add Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
git commit -m "Add Gate 1 core loop screen adaptation regressions"
```

---

### Task 2: Adapt Home Into Dashboard Regions

**Files:**
- Modify: `tldw_chatbook/UI/Screens/home_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Optional Modify: `tldw_chatbook/Home/dashboard_state.py`
- Test: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
- Test: `Tests/UI/test_home_screen.py`

- [ ] **Step 1: Preserve current Home behavior**

Before editing, run:

```bash
$PY -m pytest -q Tests/UI/test_home_screen.py --tb=short
```

Expected baseline: pass with existing warnings only. If this fails, stop and investigate the baseline before changing Home.

- [ ] **Step 2: Add Home selected-item helper**

In `HomeScreen`, store the dashboard input alongside the summarized dashboard.

```python
self._current_dashboard_input: HomeDashboardInput | None = None
```

In `compose_content()`:

```python
dashboard_input = self._build_dashboard_input()
dashboard = summarize_home_dashboard(dashboard_input)
self._current_dashboard = dashboard
self._current_dashboard_input = dashboard_input

sections = {section.section_id: section for section in dashboard.sections}

def section_text(section_id: str) -> str:
    section = sections.get(section_id)
    return "\n".join(section.lines) if section is not None else ""

attention_text = section_text("attention")
active_text = section_text("active_work")
recent_text = section_text("recent_work")
next_action_copy = f"{dashboard.next_action.label}\n{dashboard.next_action.reason}"
```

Add a private helper in `HomeScreen`:

```python
def _selected_home_item(self, dashboard_input: HomeDashboardInput):
    return dashboard_input.active_work_items[0] if dashboard_input.active_work_items else None
```

Derive selected-item copy from the same dashboard input so Home does not need to infer a target from rendered text:

```python
selected_item = self._selected_home_item(dashboard_input)
selected_item_copy = (
    f"Selected item\n{selected_item.title}\n"
    f"Source: {selected_item.source}\n"
    f"Status: {selected_item.status}\n"
    f"Target: {selected_item.detail_route}"
    if selected_item is not None
    else "Selected item\nNo active work selected."
)
```

Do not change `HomeDashboardInput` unless a pure helper is clearly useful across tests.

- [ ] **Step 3: Reshape Home composition**

Replace the single vertical report stack with contract regions while preserving existing IDs used by tests.

Update the container import first:

```python
from textual.containers import Horizontal, Vertical
```

Required region IDs:

- `#home-status-row`
- `#home-scope-filter-row`
- `#home-dashboard-grid`
- `#home-attention-queue`
- `#home-active-work-region`
- `#home-inspector`
- `#home-next-actions-region`
- `#home-recent-work-region`

Implementation shape:

```python
with Vertical(id="home-dashboard"):
    yield Static("Home", id="home-title", classes="ds-destination-header")
    yield Static(..., id="home-purpose", classes="destination-purpose")
    yield Static(
        "Home | Status, notifications, active work | Local",
        id="home-status-row",
        classes="destination-status-row",
    )
    yield Static(
        "Scope: All modules | Filter: Needs attention / Running / Recent",
        id="home-scope-filter-row",
        classes="ds-panel",
    )
    with Horizontal(id="home-dashboard-grid", classes="ds-panel"):
        with Vertical(id="home-attention-queue", classes="home-dashboard-region"):
            yield Static("Attention Queue", id="home-attention", classes="ds-panel")
            yield Static(attention_text, id="home-attention-body")
        with Vertical(id="home-active-work-region", classes="home-dashboard-region"):
            yield Static("Active Work", id="home-active-work", classes="ds-panel")
            yield Static(active_text, id="home-active-work-body")
            for control in dashboard.controls:
                yield Button(control.label, id=control.control_id, classes="ds-toolbar")
        with Vertical(id="home-inspector", classes="home-dashboard-region"):
            yield Static("Selected item", id="home-selected-item-title", classes="destination-section")
            yield Static(selected_item_copy, id="home-selected-item-body")
    with Vertical(id="home-next-actions-region", classes="ds-panel"):
        yield Static("Next Best Action", id="home-next-best-action", classes="ds-panel")
        yield Static(next_action_copy, id="home-next-best-action-body")
        yield Button(dashboard.next_action.label, id="home-primary-action")
    with Vertical(id="home-recent-work-region", classes="ds-panel"):
        yield Static("Recent Work", id="home-recent-work", classes="ds-panel")
        yield Static(recent_text, id="home-recent-work-body")
```

Use existing `dashboard.sections` values for compatibility rather than duplicating dashboard logic.

Add minimal shared layout hooks to `tldw_chatbook/css/components/_agentic_terminal.tcss` so new contract classes render as intentional panes instead of inheriting arbitrary legacy widget dimensions:

```css
.home-dashboard-region,
.console-region,
.library-region {
    width: 1fr;
    min-width: 0;
    height: auto;
}

.library-mode-chip {
    width: auto;
    min-width: 10;
}
```

- [ ] **Step 4: Run Home focused tests**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_home_core_loop_uses_dashboard_regions_and_selected_item_inspector Tests/UI/test_home_screen.py --tb=short
```

Expected: pass.

- [ ] **Step 5: Commit Home adaptation**

```bash
git add tldw_chatbook/UI/Screens/home_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/Home/dashboard_state.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
git commit -m "Adapt Home dashboard to core loop layout"
```

---

### Task 3: Adapt Console Into Agentic Shell Regions

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Optional Modify: `tldw_chatbook/Chat/console_live_work.py`
- Test: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
- Test: `Tests/UI/test_chat_first_handoffs.py`
- Test: `Tests/UI/test_chat_shell_bar.py`

- [ ] **Step 1: Preserve current Console/Chat behavior**

Run:

```bash
$PY -m pytest -q Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_shell_bar.py --tb=short
```

Expected baseline: pass with existing warnings only. If this fails, stop and investigate before changing Console.

- [ ] **Step 2: Add Console wrapper regions**

In `ChatScreen.compose_content()`, keep `ChatWindowEnhanced` intact but place it inside a new Console shell.

Update the container import first:

```python
from textual.containers import Container, Horizontal, Vertical
```

Required region IDs:

- `#console-shell`
- `#console-title`
- `#console-purpose`
- `#console-status-row`
- `#console-mode-bar`
- `#console-workspace-grid`
- `#console-staged-context-tray`
- `#console-transcript-region`
- `#console-run-inspector`
- `#console-composer-region`

Implementation shape:

```python
def compose_content(self) -> ComposeResult:
    pending_launch = self._consume_pending_console_launch()
    with Vertical(id="console-shell"):
        yield Static("Console", id="console-title", classes="ds-destination-header")
        yield Static(
            "Live agent control, chat, RAG, tools, approvals, and runs.",
            id="console-purpose",
            classes="destination-purpose",
        )
        yield Static(
            "Console | Live work, RAG, tools, approvals | Ready | Local",
            id="console-status-row",
            classes="destination-status-row",
        )
        yield Static(
            "Mode: Chat / RAG / Run Follow | Persona: Default | Sources: staged context",
            id="console-mode-bar",
            classes="ds-panel",
        )
        with Horizontal(id="console-workspace-grid", classes="ds-panel"):
            with Vertical(id="console-staged-context-tray", classes="console-region"):
                yield Static("Staged Context", classes="destination-section")
                yield Static("No staged sources yet.", id="console-staged-context-empty")
            with Vertical(id="console-transcript-region", classes="console-region"):
                yield Static("Transcript / Event Stream", classes="destination-section")
                self.chat_window = ChatWindowEnhanced(self.app_instance, id="chat-window", classes="window")
                yield self.chat_window
            with Vertical(id="console-run-inspector", classes="console-region"):
                yield Static("Run Inspector", classes="destination-section")
                if pending_launch:
                    yield from self._render_console_live_work_status_card(pending_launch)
                else:
                    yield from self._render_console_live_work_source_readiness()
        yield Static(
            "Composer: use the active chat input below. Save Chatbook from the chat controls.",
            id="console-composer-region",
            classes="ds-panel",
        )
```

Do not duplicate the real composer in this slice. The goal is to visually frame the existing chat surface.

- [ ] **Step 3: Preserve pending launch provenance**

Add a test case if needed that sets:

```python
app.pending_console_launch = {
    "source": "w+c",
    "title": "Daily papers",
    "status": "running",
    "recovery": "Follow the active watchlist run.",
    "payload": {"target_id": "local:watchlist_run:daily"},
}
```

Assert that `#console-run-inspector` contains source, title, status, recovery, and target id.

- [ ] **Step 4: Run Console focused tests**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_shell_bar.py --tb=short
```

Expected: pass.

- [ ] **Step 5: Commit Console adaptation**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_live_work.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
git commit -m "Adapt Console to agentic shell layout"
```

---

### Task 4: Make Library Modes Actionable

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
- Test: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`
- Test: `Tests/UI/test_product_maturity_phase3_knowledge_entry.py`
- Test: `Tests/UI/test_product_maturity_phase3_library_study_context.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Preserve current Library behavior**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_destination_shells.py --tb=short
```

Expected baseline: pass with existing warnings only.

- [ ] **Step 2: Add active mode state**

In `LibraryScreen.__init__()`:

```python
self._active_mode = "sources"
```

Add mode metadata:

```python
LIBRARY_MODES = (
    ("sources", "Sources", "Browse local notes, media, and conversations."),
    ("search", "Search/RAG", "Ask over selected or indexed Library sources."),
    ("import_export", "Import/Export", "Move source material in and out of Library."),
    ("workspaces", "Workspaces", "Scope Library material to a workspace."),
    ("study", "Study", "Turn sources into study sessions."),
    ("flashcards", "Flashcards", "Generate or review flashcards from Library material."),
    ("quizzes", "Quizzes", "Generate or review quizzes from Library material."),
)
```

- [ ] **Step 3: Convert mode chips to mode buttons**

Replace static mode chips with buttons that preserve the same selector IDs.

```python
for mode_id, label, tooltip in LIBRARY_MODES:
    yield Button(
        label,
        id=f"library-mode-{mode_id.replace('_', '-') if mode_id != 'search' else 'search'}",
        classes="library-mode-chip",
        tooltip=tooltip,
    )
```

Keep existing IDs used by tests:

- `#library-mode-sources`
- `#library-mode-search`
- `#library-mode-import-export`
- `#library-mode-workspaces`
- `#library-mode-study`
- `#library-mode-flashcards`
- `#library-mode-quizzes`

- [ ] **Step 4: Add mode-specific detail copy**

Inside `#library-source-detail`, render a clear mode heading before the existing source snapshot.

Examples:

```python
if self._active_mode == "search":
    yield Static("Search/RAG mode", id="library-active-mode-title", classes="destination-section")
    yield Static(
        "Ask over selected Library sources. Source evidence can be staged into Console.",
        id="library-active-mode-description",
    )
elif self._active_mode == "import_export":
    yield Static("Import/Export mode", id="library-active-mode-title", classes="destination-section")
...
else:
    yield Static("Sources mode", id="library-active-mode-title", classes="destination-section")
```

Do not implement full Search/RAG or import/export workflows in this slice.

- [ ] **Step 5: Add mode button handler**

Add `@on(Button.Pressed, ".library-mode-chip")` handler.

```python
@on(Button.Pressed, ".library-mode-chip")
def choose_library_mode(self, event: Button.Pressed) -> None:
    event.stop()
    mode_by_button_id = {
        "library-mode-sources": "sources",
        "library-mode-search": "search",
        "library-mode-import-export": "import_export",
        "library-mode-workspaces": "workspaces",
        "library-mode-study": "study",
        "library-mode-flashcards": "flashcards",
        "library-mode-quizzes": "quizzes",
    }
    next_mode = mode_by_button_id.get(event.button.id or "")
    if next_mode is None or next_mode == self._active_mode:
        return
    self._active_mode = next_mode
    self.refresh(recompose=True)
```

- [ ] **Step 6: Run Library focused tests**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_library_core_loop_modes_are_actionable_without_leaving_library Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_destination_shells.py --tb=short
```

Expected: pass.

- [ ] **Step 7: Commit Library mode adaptation**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
git commit -m "Make Library contract modes actionable"
```

---

### Task 5: Record Gate 1 QA Evidence And Tracking

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`
- Create: `backlog/tasks/task-10.4 - Product-Maturity-Phase-3.4-Core-Product-Loop-Screen-Adaptation.md`
- Modify: `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- Test: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`

- [ ] **Step 1: Add evidence-tracking test**

Extend `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py` with a durable evidence test.

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md"
)
AUDIT = Path("Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
TASK_10 = Path("backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md")
TASK_10_4 = Path(
    "backlog/tasks/task-10.4 - Product-Maturity-Phase-3.4-Core-Product-Loop-Screen-Adaptation.md"
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_gate1_core_loop_screen_adaptation_evidence_is_tracked() -> None:
    evidence = _text(EVIDENCE)
    audit = _text(AUDIT)
    tracker = _text(TRACKER)
    readme = _text(PHASE_3_README)
    task = _text(TASK_10_4)
    parent = _text(TASK_10)

    for heading in ("## Scope", "## Walkthrough", "## Verification", "## Defects", "## Exit Decision"):
        assert heading in evidence
    for selector in ("#home-dashboard-grid", "#console-workspace-grid", "#library-mode-bar"):
        assert selector in evidence
    assert EVIDENCE.name in readme
    assert EVIDENCE.name in tracker
    assert "Gate 1" in audit
    assert "TASK-10.4" in tracker
    assert "TASK-10.4" in parent
    assert "status: Done" in task
    for ac_number in range(1, 6):
        assert f"- [x] #{ac_number}" in task
    assert "## Implementation Notes" in task
```

- [ ] **Step 2: Run red evidence test**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_gate1_core_loop_screen_adaptation_evidence_is_tracked --tb=short
```

Expected before docs/tracker updates: fail because evidence and task files are missing.

- [ ] **Step 3: Add QA evidence**

Create `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md` with these sections:

- `## Scope`
- `## Walkthrough`
- `## Functional Result`
- `## Verification`
- `## Defects`
- `## Residual Risk`
- `## Exit Decision`

Record:

- Home dashboard regions verified.
- Console shell regions verified.
- Library actionable mode selection verified.
- Focused pytest commands and results.
- Known residual risk that `ChatWindowEnhanced` remains legacy internals inside the Console shell.

- [ ] **Step 4: Update roadmap and QA index**

Update:

- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`

Use stable references:

- `TASK-10.4`
- evidence filename
- selectors `#home-dashboard-grid`, `#console-workspace-grid`, `#library-mode-bar`

- [ ] **Step 5: Add Backlog task hygiene**

Create `backlog/tasks/task-10.4 - Product-Maturity-Phase-3.4-Core-Product-Loop-Screen-Adaptation.md`.

Minimum acceptance criteria:

- `#1` Home exposes dashboard regions and selected-item inspector.
- `#2` Console exposes staged context, transcript, inspector, and composer contract regions.
- `#3` Library modes are actionable without leaving Library.
- `#4` Mounted UI regressions cover Home, Console, and Library.
- `#5` QA walkthrough verifies the app is usable, not merely clickable.

Mark each criterion checked only after the implementation and verification are complete. Add `## Implementation Notes` with concise PR-style summary.

- [ ] **Step 6: Run evidence test green**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_gate1_core_loop_screen_adaptation_evidence_is_tracked --tb=short
```

Expected: pass.

- [ ] **Step 7: Commit docs and tracking**

```bash
git add Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md Docs/superpowers/qa/product-maturity/phase-3/README.md Docs/superpowers/trackers/product-maturity-roadmap.md Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md "backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md" "backlog/tasks/task-10.4 - Product-Maturity-Phase-3.4-Core-Product-Loop-Screen-Adaptation.md" Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
git commit -m "Record Gate 1 core loop screen adaptation evidence"
```

---

### Task 6: Final Verification And PR Prep

**Files:**
- Read: all modified files
- Verify: focused UI tests and diff hygiene

- [ ] **Step 1: Run full focused verification**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_home_screen.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_shell_bar.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_destination_shells.py Tests/UI/test_master_shell_design_system_contract.py --tb=short
```

Expected: all selected tests pass with only known dependency warnings.

- [ ] **Step 2: Run formatting/diff hygiene**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; branch shows only intended committed changes or a clean tree.

- [ ] **Step 3: Self-review against audit**

Re-read:

```bash
sed -n '1,220p' Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
```

Confirm:

- Gate 1 did not accidentally expand into Gate 2 or Gate 3.
- Gate 1.5 and Gate 1.6 remain explicitly documented as required follow-up gates rather than optional cleanup.
- `Console` still uses route id `chat`.
- `Library` still owns Study, Flashcards, Quizzes, Search/RAG, Import/Export, Notes, Media, and Conversations.
- `Home` controls still route through existing adapter methods.

- [ ] **Step 4: Prepare PR summary**

Use this structure:

```markdown
## Summary
- Adapted Home into dashboard/control regions with selected-item inspector.
- Adapted Console into an agentic shell around the existing chat surface.
- Made Library contract modes actionable while preserving source snapshot and handoff behavior.
- Added Gate 1 mounted regressions and QA evidence.

## Verification
- `$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py ... --tb=short`
- `git diff --check`
```

- [ ] **Step 5: Commit any final fixes**

If final verification required fixes:

```bash
git add <changed-files>
git commit -m "Stabilize Gate 1 core loop screen adaptation"
```

Expected: branch is ready for PR against `dev`.
