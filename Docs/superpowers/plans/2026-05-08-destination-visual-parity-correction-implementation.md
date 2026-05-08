# Destination Visual Parity Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring every top-level destination screen into visible layout parity with the approved ASCII contracts at normal terminal sizes without pretending missing backend depth is implemented.

**Architecture:** Add a shared destination workbench grammar, then apply it across screens in small slices. Preserve existing services, routes, stable selectors, and Console handoff behavior; only reflow the visible Textual layout and honest recovery states.

**Tech Stack:** Python 3.11+, Textual, pytest, TCSS source modules regenerated through `tldw_chatbook/css/build_css.py`.

---

## Spec And Guardrails

Read these first:

- `Docs/superpowers/specs/2026-05-08-destination-visual-parity-correction-design.md`
- `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- `Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md`
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- `AGENTS.md`

Hard rules:

- Do not implement backend feature depth that does not already exist.
- Do not move Collections back into Watchlists.
- Do not make ACP, MCP, Skills, Schedules, Workflows, or Settings look more functional than they are.
- Do not manually edit `tldw_chatbook/css/tldw_cli_modular.tcss`; edit source TCSS and regenerate it.
- Preserve route IDs and existing stable selectors unless a test is intentionally migrated with compatibility coverage.
- Do not call a slice done because selectors exist. Each slice must assert rendered geometry: pane ordering, viewport visibility, and compact header/mode-bar height at `140x42`.
- Do not use fixed `pilot.pause(...)` sleeps for async screen readiness when a deterministic selector/state wait exists. Reuse `_wait_for_selector(...)` and existing destination-specific wait helpers.

Use this Python unless the environment changes:

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

## File Structure

Create:

- `Tests/UI/test_destination_visual_parity_correction.py`
  - Mounted geometry and focus-order tests for the correction pass.
- `tldw_chatbook/Widgets/destination_workbench.py`
  - Small reusable pane widgets/helpers for list/detail/inspector workbench layouts.

Modify:

- `tldw_chatbook/UI/Navigation/base_app_screen.py`
  - Remove duplicate vertical offset and keep screen content immediately below docked nav.
- `tldw_chatbook/UI/Navigation/main_navigation.py`
  - Prevent `More: Ctrl+P` overlap and preserve Home/Console visibility.
- `tldw_chatbook/UI/Screens/home_screen.py`
  - Compress Home into one visible dashboard.
- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Reflow Console into staged-context / transcript / inspector workbench.
- `tldw_chatbook/Widgets/Console/console_control_bar.py`
- `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- `tldw_chatbook/Widgets/Console/console_run_inspector.py`
- `tldw_chatbook/Widgets/Console/console_staged_context.py`
- `tldw_chatbook/Widgets/Console/console_session_surface.py`
  - Keep Console widgets compact and usable inside the corrected workbench.
- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/Widgets/Library/library_collections_panel.py`
  - Make Library mode strip compact and keep Library subflows inside source/detail/inspector geometry.
- `tldw_chatbook/UI/Screens/artifacts_screen.py`
- `tldw_chatbook/UI/Screens/personas_screen.py`
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- `tldw_chatbook/UI/Screens/skills_screen.py`
  - Convert source-prep/output screens from vertical stacks into list/detail/inspector workbenches.
- `tldw_chatbook/UI/Screens/schedules_screen.py`
- `tldw_chatbook/UI/Screens/workflows_screen.py`
  - Convert operational placeholder stacks into timing/procedure workbenches.
- `tldw_chatbook/UI/Screens/mcp_screen.py`
- `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
  - Add a visual adapter or compact mode around existing MCP state/action seams.
- `tldw_chatbook/UI/Screens/acp_screen.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Convert runtime/config shells into pane layouts with honest unavailable states.
- `tldw_chatbook/css/core/_variables.tcss`
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
- `tldw_chatbook/css/components/_navigation.tcss`
- `tldw_chatbook/css/layout/_panes.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Source TCSS changes plus generated CSS output.
- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
  - QA/tracking closeout after verification.

---

### Task 1: Geometry Harness And Shell/Nav Baseline

**Files:**
- Create: `Tests/UI/test_destination_visual_parity_correction.py`
- Modify: `tldw_chatbook/UI/Navigation/base_app_screen.py`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/css/components/_navigation.tcss`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/core/_variables.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing geometry helpers and nav tests**

Add helpers and the first tests:

```python
"""Visual parity geometry tests for destination correction pass."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _wait_for_selector,
)
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar


def _region(widget):
    region = widget.region
    return region.x, region.y, region.width, region.height


def _bottom(widget) -> int:
    x, y, width, height = _region(widget)
    return y + height


def _assert_visible_in_viewport(widget, *, height: int, context: str) -> None:
    x, y, width, widget_height = _region(widget)
    assert y >= 0, context
    assert y < height, context
    assert y + widget_height <= height, context


def _assert_no_horizontal_overlap(left, right, *, context: str) -> None:
    lx, ly, lw, lh = _region(left)
    rx, ry, rw, rh = _region(right)
    if ly + lh <= ry or ry + rh <= ly:
        return
    assert lx + lw <= rx or rx + rw <= lx, context


@pytest.mark.asyncio
async def test_main_navigation_overflow_hint_does_not_overlap_settings_at_default_size():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-dashboard")
        nav = host.query_one(MainNavigationBar)
        settings = nav.query_one("#nav-settings", Button)
        more = nav.query_one("#nav-overflow-hint")
        _assert_no_horizontal_overlap(settings, more, context="More hint overlaps Settings nav item")


@pytest.mark.asyncio
async def test_destination_content_starts_immediately_below_nav():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-dashboard")
        content = host.query_one("#screen-content")
        dashboard = home.query_one("#home-dashboard")
        assert content.region.y == 3
        assert dashboard.region.y <= 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_main_navigation_overflow_hint_does_not_overlap_settings_at_default_size Tests/UI/test_destination_visual_parity_correction.py::test_destination_content_starts_immediately_below_nav --tb=short
```

Expected: FAIL because `More: Ctrl+P` overlaps Settings and content starts too low.

- [ ] **Step 3: Fix base screen vertical offset**

In `tldw_chatbook/UI/Navigation/base_app_screen.py`, change `#screen-content` so it does not add a second nav offset:

```python
DEFAULT_CSS = """
BaseAppScreen {
    background: $background;
}

#screen-content {
    width: 100%;
    height: 100%;
    padding-top: 0;
}
"""
```

- [ ] **Step 4: Fix nav overflow hint layout**

In `tldw_chatbook/UI/Navigation/main_navigation.py`, split destination buttons from the hint so the hint can dock right:

```python
def compose(self) -> ComposeResult:
    yield Static("More: Ctrl+P", id="nav-overflow-hint", classes="nav-overflow-hint")
    with Horizontal(id="nav-destination-strip", classes="main-nav"):
        for index, destination in enumerate(SHELL_DESTINATION_ORDER):
            ...
```

Update `DEFAULT_CSS` so `#nav-destination-strip` leaves room for the docked hint:

```css
MainNavigationBar {
    height: 3;
    width: 100%;
    dock: top;
    background: $panel;
    border-bottom: solid $primary;
    overflow: hidden;
}

#nav-overflow-hint {
    dock: right;
    width: 14;
    height: 1;
    padding: 0 1;
    color: $text-muted;
}

#nav-destination-strip {
    height: 3;
    width: 1fr;
    margin-right: 14;
    overflow-x: auto;
}
```

- [ ] **Step 5: Add source TCSS equivalents**

Mirror the navigation CSS in `tldw_chatbook/css/components/_navigation.tcss` if the app-level defaults are not sufficient. Do not edit generated CSS manually.

- [ ] **Step 6: Regenerate CSS**

Run:

```bash
$PY tldw_chatbook/css/build_css.py
```

Expected: exits 0. The pre-existing missing `features/_evaluation_v2.tcss` warning is acceptable if unchanged.

- [ ] **Step 7: Verify task tests pass**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_main_navigation_overflow_hint_does_not_overlap_settings_at_default_size Tests/UI/test_destination_visual_parity_correction.py::test_destination_content_starts_immediately_below_nav --tb=short
git diff --check
```

Expected: PASS and no whitespace errors.

- [ ] **Step 8: Commit**

```bash
git add Tests/UI/test_destination_visual_parity_correction.py tldw_chatbook/UI/Navigation/base_app_screen.py tldw_chatbook/UI/Navigation/main_navigation.py tldw_chatbook/css/components/_navigation.tcss tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/core/_variables.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Fix shell navigation visual density"
```

---

### Task 2: Shared Destination Workbench Primitives

**Files:**
- Create: `tldw_chatbook/Widgets/destination_workbench.py`
- Modify: `tldw_chatbook/Widgets/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/layout/_panes.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_destination_visual_parity_correction.py`

- [ ] **Step 1: Write failing helper widget tests**

Add a lightweight mounted app in `Tests/UI/test_destination_visual_parity_correction.py` that mounts `DestinationWorkbench` with three panes and asserts pane ordering:

```python
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Widgets.destination_workbench import DestinationWorkbench, WorkbenchPane


class WorkbenchHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield DestinationWorkbench(
            WorkbenchPane("List", Static("left"), id="test-list-pane"),
            WorkbenchPane("Detail", Static("center"), id="test-detail-pane"),
            WorkbenchPane("Inspector", Static("right"), id="test-inspector-pane"),
            id="test-workbench",
        )


@pytest.mark.asyncio
async def test_destination_workbench_renders_three_horizontal_panes():
    app = WorkbenchHarness()
    async with app.run_test(size=(100, 20)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#test-workbench")
        left = app.query_one("#test-list-pane")
        center = app.query_one("#test-detail-pane")
        right = app.query_one("#test-inspector-pane")
        assert left.region.x < center.region.x < right.region.x
        assert left.region.y == center.region.y == right.region.y
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_destination_workbench_renders_three_horizontal_panes --tb=short
```

Expected: FAIL because `destination_workbench.py` does not exist.

- [ ] **Step 3: Implement minimal reusable workbench widgets**

Create `tldw_chatbook/Widgets/destination_workbench.py`:

```python
"""Reusable destination workbench panes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static


@dataclass(frozen=True)
class WorkbenchPane:
    """A titled pane in a destination workbench."""

    title: str
    content: Widget | Iterable[Widget]
    id: str
    classes: str = ""


class DestinationWorkbench(Horizontal):
    """Three-pane terminal-native destination workbench."""

    def __init__(self, *panes: WorkbenchPane, **kwargs) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"destination-workbench {classes}".strip(), **kwargs)
        self.panes = panes

    def compose(self) -> ComposeResult:
        for pane in self.panes:
            with Vertical(id=pane.id, classes=f"destination-workbench-pane {pane.classes}".strip()):
                yield Static(pane.title, classes="destination-pane-title")
                content = pane.content
                if isinstance(content, Widget):
                    yield content
                else:
                    yield from content
```

Export it in `tldw_chatbook/Widgets/__init__.py` if that file already exports shared widgets. If it is empty or not used, direct imports are acceptable.

- [ ] **Step 4: Add TCSS source rules**

Add to `tldw_chatbook/css/components/_agentic_terminal.tcss` or `layout/_panes.tcss`:

```css
.destination-workbench {
    height: 1fr;
    min-height: 0;
    width: 100%;
}

.destination-workbench-pane {
    width: 1fr;
    min-width: 0;
    height: 100%;
    min-height: 0;
    padding: 0 1;
    border: round $ds-grid-line;
    background: $ds-surface-panel;
}

.destination-pane-title {
    height: 1;
    min-height: 1;
    text-style: bold;
    color: $ds-text-primary;
}

.destination-mode-strip,
.destination-filter-strip,
.destination-footer-row {
    height: 1;
    min-height: 1;
    padding: 0 1;
}
```

- [ ] **Step 5: Regenerate CSS**

Run:

```bash
$PY tldw_chatbook/css/build_css.py
```

- [ ] **Step 6: Verify tests pass**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_destination_workbench_renders_three_horizontal_panes --tb=short
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add Tests/UI/test_destination_visual_parity_correction.py tldw_chatbook/Widgets/destination_workbench.py tldw_chatbook/Widgets/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/layout/_panes.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Add destination workbench layout primitives"
```

---

### Task 3: Home, Console, And Library Visual Parity

**Files:**
- Modify: `Tests/UI/test_destination_visual_parity_correction.py`
- Modify: `tldw_chatbook/UI/Screens/home_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/Widgets/Console/console_run_inspector.py`
- Modify: `tldw_chatbook/Widgets/Console/console_staged_context.py`
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_collections_panel.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing Home/Console/Library geometry tests**

Add tests:

```python
@pytest.mark.asyncio
async def test_home_dashboard_regions_fit_default_viewport():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-dashboard-grid")
        for selector in (
            "#home-dashboard-grid",
            "#home-next-actions-region",
            "#home-recent-work-region",
        ):
            _assert_visible_in_viewport(home.query_one(selector), height=42, context=selector)


@pytest.mark.asyncio
async def test_console_uses_three_pane_workbench_and_visible_composer():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")
        context = console.query_one("#console-staged-context-tray")
        transcript = console.query_one("#console-session-surface")
        inspector = console.query_one("#console-run-inspector")
        composer = console.query_one("#console-native-composer")
        assert context.region.x < transcript.region.x < inspector.region.x
        assert context.region.y == transcript.region.y == inspector.region.y
        _assert_visible_in_viewport(composer, height=42, context="Console composer")


@pytest.mark.asyncio
async def test_library_mode_strip_is_compact_and_workbench_visible():
    app = _build_test_app()
    host = DestinationHarness(app, "library")
    async with host.run_test(size=(140, 42)) as pilot:
        library = _active_destination_screen(host)
        await _wait_for_selector(library, pilot, "#library-contract-grid")
        assert library.query_one("#library-mode-bar").region.height <= 2
        for selector in (
            "#library-source-browser",
            "#library-source-detail",
            "#library-source-inspector",
        ):
            _assert_visible_in_viewport(library.query_one(selector), height=42, context=selector)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_home_dashboard_regions_fit_default_viewport Tests/UI/test_destination_visual_parity_correction.py::test_console_uses_three_pane_workbench_and_visible_composer Tests/UI/test_destination_visual_parity_correction.py::test_library_mode_strip_is_compact_and_workbench_visible --tb=short
```

- [ ] **Step 3: Reflow Home**

In `home_screen.py`:

- Keep `#home-dashboard-grid`, `#home-attention-queue`, `#home-active-work-region`, and `#home-inspector`.
- Move `#home-next-actions-region` and `#home-recent-work-region` into a compact bottom row under the grid.
- Keep existing control IDs, including `home-open-chatbook-details` and `home-open-chatbook-in-console`.

Use `Horizontal(id="home-followup-row")` for next/recent:

```python
with Horizontal(id="home-followup-row", classes="ds-panel destination-footer-row"):
    with Vertical(id="home-next-actions-region"):
        ...
    with Vertical(id="home-recent-work-region"):
        ...
```

- [ ] **Step 4: Reflow Console**

In `chat_screen.py`, change the workspace from left stacked context/inspector plus right transcript/composer into three panes:

```python
yield ConsoleControlBar(..., id="console-control-bar", classes="destination-mode-strip")
with Horizontal(id="console-workspace-grid", classes="destination-workbench ds-panel"):
    yield ConsoleStagedContextTray(..., id="console-staged-context-tray", classes="console-region")
    with Vertical(id="console-main-column", classes="console-region console-transcript-pane"):
        with Vertical(id="console-transcript-region", classes="console-region"):
            yield self._ensure_console_session_surface()
        yield ConsoleComposerBar(id="console-native-composer", classes="console-region destination-footer-row")
    with Vertical(id="console-run-inspector", classes="console-region ds-inspector"):
        yield ConsoleRunInspector(...)
        ...
```

Compress child widgets with TCSS before changing state logic.

- [ ] **Step 5: Reflow Library**

In `library_screen.py`:

- Keep `#library-mode-bar`, but make it a compact strip.
- Ensure `#library-contract-grid` starts immediately after the mode strip.
- In Collections mode, render `LibraryCollectionsPanel` inside `#library-source-detail` and keep collection actions/status in `#library-source-inspector`.

In `library_collections_panel.py`, remove nested tall form blocks where possible and keep list/detail visible above action rows.

- [ ] **Step 6: Update TCSS source and regenerate CSS**

Adjust only source TCSS. Run:

```bash
$PY tldw_chatbook/css/build_css.py
```

- [ ] **Step 7: Run focused verification**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase39_library_collections.py --tb=short
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add Tests/UI/test_destination_visual_parity_correction.py tldw_chatbook/UI/Screens/home_screen.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Reflow core destinations to visual contracts"
```

---

### Task 4: Artifacts, Personas, Watchlists, And Skills Workbenches

**Files:**
- Modify: `Tests/UI/test_destination_visual_parity_correction.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/UI/Screens/skills_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing workbench geometry tests**

Add parametrized tests:

```python
SOURCE_PREP_WORKBENCHES = {
    "artifacts": ("#artifacts-workbench", "#artifacts-list-pane", "#artifacts-detail-pane", "#artifacts-inspector-pane"),
    "personas": ("#personas-workbench", "#personas-list-pane", "#personas-detail-pane", "#personas-inspector-pane"),
    "watchlists_collections": ("#watchlists-workbench", "#watchlists-list-pane", "#watchlists-detail-pane", "#watchlists-inspector-pane"),
    "skills": ("#skills-workbench", "#skills-list-pane", "#skills-detail-pane", "#skills-inspector-pane"),
}


@pytest.mark.parametrize("route,selectors", SOURCE_PREP_WORKBENCHES.items())
@pytest.mark.asyncio
async def test_source_prep_destinations_use_list_detail_inspector_workbench(route, selectors):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, selectors[0])
        panes = [screen.query_one(selector) for selector in selectors[1:]]
        assert panes[0].region.x < panes[1].region.x < panes[2].region.x
        for pane in panes:
            _assert_visible_in_viewport(pane, height=42, context=f"{route}:{pane.id}")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_source_prep_destinations_use_list_detail_inspector_workbench --tb=short
```

- [ ] **Step 3: Convert Artifacts**

In `artifacts_screen.py`, introduce:

- `#artifacts-mode-strip`
- `#artifacts-workbench`
- `#artifacts-list-pane`
- `#artifacts-detail-pane`
- `#artifacts-inspector-pane`

Keep existing IDs such as `#artifacts-open-chatbooks`, `#artifacts-console-available`, and `#artifacts-console-unavailable` inside the relevant panes.

- [ ] **Step 4: Convert Personas**

In `personas_screen.py`, introduce:

- `#personas-mode-strip`
- `#personas-workbench`
- `#personas-list-pane`
- `#personas-detail-pane`
- `#personas-inspector-pane`

Render current local character/profile snapshot rows in the list pane. Put `Open Personas` and `Attach to Console` in the inspector pane.

- [ ] **Step 5: Convert Watchlists**

In `watchlists_collections_screen.py`, introduce:

- `#watchlists-filter-strip`
- `#watchlists-workbench`
- `#watchlists-list-pane`
- `#watchlists-detail-pane`
- `#watchlists-inspector-pane`

Use Watchlists in all user-facing labels. Preserve legacy route id `watchlists_collections` and existing button IDs. Do not show Collections as managed here.

- [ ] **Step 6: Convert Skills**

In `skills_screen.py`, introduce:

- `#skills-mode-strip`
- `#skills-workbench`
- `#skills-list-pane`
- `#skills-detail-pane`
- `#skills-inspector-pane`

Render local skill names in the list pane, `SKILL.md` metadata placeholders in detail, and validation/import/attach actions in inspector.

- [ ] **Step 7: Regenerate CSS and verify**

Run:

```bash
$PY tldw_chatbook/css/build_css.py
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_source_prep_destinations_use_list_detail_inspector_workbench Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py --tb=short
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add Tests/UI/test_destination_visual_parity_correction.py tldw_chatbook/UI/Screens/artifacts_screen.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/UI/Screens/skills_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Convert source prep destinations to workbench layouts"
```

---

### Task 5: Schedules And Workflows Workbenches

**Files:**
- Modify: `Tests/UI/test_destination_visual_parity_correction.py`
- Modify: `tldw_chatbook/UI/Screens/schedules_screen.py`
- Modify: `tldw_chatbook/UI/Screens/workflows_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing operational geometry tests**

Add:

```python
@pytest.mark.parametrize(
    "route,workbench,list_pane,detail_pane,inspector_pane",
    [
        ("schedules", "#schedules-workbench", "#schedules-list-pane", "#schedules-detail-pane", "#schedules-inspector-pane"),
        ("workflows", "#workflows-workbench", "#workflows-list-pane", "#workflows-detail-pane", "#workflows-inspector-pane"),
    ],
)
@pytest.mark.asyncio
async def test_operational_destinations_use_timing_or_procedure_workbench(
    route, workbench, list_pane, detail_pane, inspector_pane
):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        panes = [screen.query_one(list_pane), screen.query_one(detail_pane), screen.query_one(inspector_pane)]
        assert panes[0].region.x < panes[1].region.x < panes[2].region.x
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_operational_destinations_use_timing_or_procedure_workbench --tb=short
```

- [ ] **Step 3: Convert Schedules**

In `schedules_screen.py`, introduce:

- `#schedules-filter-strip`
- `#schedules-workbench`
- `#schedules-list-pane`
- `#schedules-detail-pane`
- `#schedules-inspector-pane`
- `#schedules-history-row`

Keep `#schedules-console-available`, `#schedules-console-unavailable`, and `#schedules-follow-in-console`.

- [ ] **Step 4: Convert Workflows**

In `workflows_screen.py`, introduce:

- `#workflows-mode-strip`
- `#workflows-workbench`
- `#workflows-list-pane`
- `#workflows-detail-pane`
- `#workflows-inspector-pane`

Keep `#workflows-console-available`, `#workflows-console-unavailable`, and `#workflows-launch-in-console`.

- [ ] **Step 5: Regenerate CSS and verify**

Run:

```bash
$PY tldw_chatbook/css/build_css.py
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_operational_destinations_use_timing_or_procedure_workbench Tests/UI/test_console_live_work_handoffs.py --tb=short
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add Tests/UI/test_destination_visual_parity_correction.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Convert operational destinations to workbench layouts"
```

---

### Task 6: MCP Visual Adapter And Overflow Correction

**Files:**
- Modify: `Tests/UI/test_destination_visual_parity_correction.py`
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing MCP overflow tests**

Add:

```python
@pytest.mark.asyncio
async def test_mcp_uses_visible_server_detail_readiness_layout_without_overflow():
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#mcp-workbench")
        for selector in (
            "#mcp-server-tree-pane",
            "#mcp-detail-pane",
            "#mcp-readiness-pane",
            "#unified-mcp-action-run",
        ):
            _assert_visible_in_viewport(screen.query_one(selector), height=42, context=selector)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_mcp_uses_visible_server_detail_readiness_layout_without_overflow --tb=short
```

- [ ] **Step 3: Add MCP screen-level workbench wrapper**

In `mcp_screen.py`, create the destination shell:

- `#mcp-mode-strip`
- `#mcp-workbench`
- `#mcp-server-tree-pane`
- `#mcp-detail-pane`
- `#mcp-readiness-pane`

Mount `UnifiedMCPPanel` in compact/detail mode if possible.

- [ ] **Step 4: Add compact mode to `UnifiedMCPPanel`**

Avoid rewriting service/action logic. Add optional `layout_mode: str = "full"` to `UnifiedMCPPanel.__init__`.

For `layout_mode == "compact-workbench"`:

- Render Source/Server/Scope/Section as a compact strip or concise rows.
- Render active section content in the detail pane.
- Render action payload/result controls in the readiness pane.
- Preserve existing IDs: `#unified-mcp-source`, `#unified-mcp-server-target`, `#unified-mcp-section`, `#unified-mcp-action`, `#unified-mcp-action-payload`, `#unified-mcp-action-run`.

- [ ] **Step 5: Verify MCP behavior tests still pass**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_mcp_uses_visible_server_detail_readiness_layout_without_overflow Tests/UI/test_destination_shells.py::test_mcp_destination_embeds_unified_mcp_management_panel Tests/UI/test_destination_shells.py::test_mcp_destination_restores_unified_mcp_view_state_after_mount Tests/UI/test_destination_shells.py::test_mcp_destination_runtime_refresh_uses_exclusive_worker --tb=short
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add Tests/UI/test_destination_visual_parity_correction.py tldw_chatbook/UI/Screens/mcp_screen.py tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Adapt MCP destination to visual workbench"
```

---

### Task 7: ACP And Settings Workbenches

**Files:**
- Modify: `Tests/UI/test_destination_visual_parity_correction.py`
- Modify: `tldw_chatbook/UI/Screens/acp_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing runtime/config geometry tests**

Add:

```python
@pytest.mark.parametrize(
    "route,workbench,list_pane,detail_pane,inspector_pane",
    [
        ("acp", "#acp-workbench", "#acp-list-pane", "#acp-detail-pane", "#acp-inspector-pane"),
        ("settings", "#settings-workbench", "#settings-category-pane", "#settings-detail-pane", "#settings-impact-pane"),
    ],
)
@pytest.mark.asyncio
async def test_runtime_and_settings_destinations_use_pane_layouts(
    route, workbench, list_pane, detail_pane, inspector_pane
):
    app = _build_test_app()
    host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, workbench)
        panes = [screen.query_one(list_pane), screen.query_one(detail_pane), screen.query_one(inspector_pane)]
        assert panes[0].region.x < panes[1].region.x < panes[2].region.x
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_destinations_use_pane_layouts --tb=short
```

- [ ] **Step 3: Convert ACP**

In `acp_screen.py`, introduce:

- `#acp-mode-strip`
- `#acp-workbench`
- `#acp-list-pane`
- `#acp-detail-pane`
- `#acp-inspector-pane`

Keep `#acp-empty-state`, `#acp-console-unavailable`, `#acp-follow-in-console`, and `#acp-launch-agent`.

- [ ] **Step 4: Convert Settings**

In `settings_screen.py`, introduce:

- `#settings-category-strip`
- `#settings-workbench`
- `#settings-category-pane`
- `#settings-detail-pane`
- `#settings-impact-pane`

Keep `#settings-open-appearance` and boundary copy.

- [ ] **Step 5: Verify tests**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_destinations_use_pane_layouts Tests/UI/test_destination_shells.py --tb=short
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add Tests/UI/test_destination_visual_parity_correction.py tldw_chatbook/UI/Screens/acp_screen.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Convert runtime and settings destinations to workbench layouts"
```

---

### Task 8: Compact Size, Focus Order, And Full QA Closeout

**Files:**
- Modify: `Tests/UI/test_destination_visual_parity_correction.py`
- Modify: destination files as needed from previous tasks
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- Create: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-destination-visual-parity-correction.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`

- [ ] **Step 1: Add compact `100x32` regression**

Add:

```python
TOP_LEVEL_WORKBENCH_SELECTORS = {
    "home": "#home-dashboard-grid",
    "chat": "#console-workspace-grid",
    "library": "#library-contract-grid",
    "artifacts": "#artifacts-workbench",
    "personas": "#personas-workbench",
    "watchlists_collections": "#watchlists-workbench",
    "schedules": "#schedules-workbench",
    "workflows": "#workflows-workbench",
    "mcp": "#mcp-workbench",
    "acp": "#acp-workbench",
    "skills": "#skills-workbench",
    "settings": "#settings-workbench",
}


@pytest.mark.parametrize("route,workbench", TOP_LEVEL_WORKBENCH_SELECTORS.items())
@pytest.mark.asyncio
async def test_top_level_destinations_keep_primary_workbench_visible_at_compact_size(route, workbench):
    app = _build_test_app()
    if route == "home":
        host = HomeHarness(app)
    elif route == "chat":
        host = ConsoleHarness(app)
    else:
        host = DestinationHarness(app, route)
    async with host.run_test(size=(100, 32)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, workbench)
        _assert_visible_in_viewport(screen.query_one(workbench), height=32, context=f"{route}:{workbench}")
```

- [ ] **Step 2: Add focus-order smoke test**

Add a smoke-level test that proves Tab reaches at least one visible primary control for each destination. This catches layouts where focusable controls exist but are hidden below the viewport.

```python
VISIBLE_FOCUS_TARGETS = {
    "home": {"home-primary-action", "home-open-details", "home-open-in-console", "home-open-chatbook-details"},
    "chat": {"console-send-message", "console-attach-context", "console-save-chatbook", "console-run-library-rag"},
    "library": {"library-open-notes", "library-open-media", "library-open-search", "library-use-in-console"},
    "artifacts": {"artifacts-open-chatbooks", "artifacts-use-in-console"},
    "personas": {"personas-open-profiles", "personas-attach-to-console"},
    "watchlists_collections": {"wc-open-watchlists", "wc-attach-to-console", "watchlists-follow-in-console"},
    "schedules": {"schedules-follow-in-console"},
    "workflows": {"workflows-launch-in-console"},
    "mcp": {"unified-mcp-action-run"},
    "acp": {"acp-follow-in-console", "acp-launch-agent"},
    "skills": {"skills-import-skill", "skills-attach-to-console"},
    "settings": {"settings-open-appearance"},
}


@pytest.mark.parametrize("route,targets", VISIBLE_FOCUS_TARGETS.items())
@pytest.mark.asyncio
async def test_tab_order_reaches_visible_primary_action(route, targets):
    app = _build_test_app()
    if route == "home":
        host = HomeHarness(app)
    elif route == "chat":
        host = ConsoleHarness(app)
    else:
        host = DestinationHarness(app, route)
    async with host.run_test(size=(140, 42)) as pilot:
        workbench = TOP_LEVEL_WORKBENCH_SELECTORS[route]
        await _wait_for_selector(host.screen_stack[-1], pilot, workbench)
        for _ in range(24):
            await pilot.press("tab")
            focused = host.focused
            if focused is not None and focused.id in targets:
                _assert_visible_in_viewport(
                    focused,
                    height=42,
                    context=f"{route}:{focused.id} focused below viewport",
                )
                return
        pytest.fail(f"{route} did not focus a visible primary action from {sorted(targets)}")
```

Keep this smoke-level; do not overfit every widget or require an exact full-screen focus sequence.

- [ ] **Step 3: Run compact tests and fix remaining geometry**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py --tb=short
```

Fix any screens that still overflow or hide primary actions.

- [ ] **Step 4: Run focused UI suites**

Run:

```bash
$PY -m pytest -q \
  Tests/UI/test_destination_visual_parity_correction.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/UI/test_product_maturity_phase39_library_collections.py \
  Tests/UI/test_screen_navigation.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 5: Run CSS build and diff hygiene**

Run:

```bash
$PY tldw_chatbook/css/build_css.py
git diff --check
```

Expected: CSS build exits 0 and diff check passes.

- [ ] **Step 6: Write QA evidence**

Create `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-destination-visual-parity-correction.md` with:

- Commit/branch under test.
- Terminal sizes checked: `140x42`, `100x32`.
- List of all top-level destinations verified.
- Known residual risks, if any.
- Exact pytest commands and results.
- CSS build result and any unchanged pre-existing warnings.

- [ ] **Step 7: Update roadmap/backlog tracking**

Update:

- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`

Do not mark Phase 3 fully complete unless remaining Phase 3 feature-depth risks are actually done. Mark this as visual parity correction verified.

- [ ] **Step 8: Final verification**

Run:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py --tb=short
git diff --check
git status --short
```

- [ ] **Step 9: Commit**

```bash
git add Tests/UI/test_destination_visual_parity_correction.py Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-destination-visual-parity-correction.md Docs/superpowers/qa/product-maturity/phase-3/README.md Docs/superpowers/trackers/product-maturity-roadmap.md "backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md"
git commit -m "Verify destination visual parity correction"
```

---

## Final PR Preparation

After all tasks:

- [ ] Run final focused verification:

```bash
$PY -m pytest -q \
  Tests/UI/test_destination_visual_parity_correction.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/UI/test_product_maturity_phase39_library_collections.py \
  Tests/UI/test_screen_navigation.py \
  --tb=short
git diff --check
```

- [ ] Push branch and open PR against `dev`.
- [ ] In the PR body, include:
  - Design spec link.
  - QA evidence link.
  - Exact verification commands.
  - Note that backend feature depth is intentionally unchanged.
  - Note that Collections remain Library-owned and Watchlists is top-level.
