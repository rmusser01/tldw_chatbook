# Chatbook Workbench UI Foundation And Console Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the measured Workbench UI foundation and use Console as the first Posting-inspired reference implementation without hiding core workflows behind the command palette.

**Architecture:** Add a small Textual-native workbench layer that owns shared layout primitives, state snapshots, focus/help contracts, and responsiveness diagnostics while domain screens continue to own domain data and side effects. Console is refactored to consume those primitives first because it exercises streaming, provider readiness, staged context, rails, transcript updates, and recovery states. Later destination migrations use separate plans and must pass the route-owner and responsiveness gates introduced here.

**Tech Stack:** Python 3.11, Textual 3.3+, Rich, pytest, pytest-asyncio, existing TCSS build pipeline, Backlog.md task `TASK-141`, ADR-011.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-06-29-posting-inspired-chatbook-ui-redesign-design.md`
- Backlog task: `backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md`
- ADR: `backlog/decisions/011-chatbook-workbench-ui-system.md`
- Posting reference checkout: `/tmp/posting-main`
- Current Console screen: `tldw_chatbook/UI/Screens/chat_screen.py`
- Current Console widgets: `tldw_chatbook/Widgets/Console/`
- Current route sources: `tldw_chatbook/Constants.py`, `tldw_chatbook/UI/Navigation/shell_destinations.py`, `tldw_chatbook/UI/Navigation/screen_registry.py`

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`

Reason: this changes long-lived UI structure, shared Textual widget contracts, route migration ownership, focus/help conventions, and responsiveness gates for the redesign.

## Scope Boundary

This plan implements the foundation plus Console reference slice. It does not migrate Library, Notes, Personas, Watchlists, Schedules, Workflows, MCP, ACP, Skills, Settings, Artifacts, Home, or legacy route screens beyond route-coverage tracking. Those destination migrations need separate tasks/plans after this foundation lands.

Core Console workflows must be visible in the workbench itself. The command palette may duplicate actions, but it must not be the only path for provider/model setup, send/stop, attach context, Library RAG, inspector review, save Chatbook, rail open/close, or recovery actions.

## Posting Practices To Carry Forward

- Compose a stable widget tree once and sync state into existing widgets.
- Prefer reactive classes, `set_class`, and explicit `sync_state()` methods over remounting entire regions.
- Use lazy or deferred heavy panes only at route/pane boundaries.
- Push typed widget messages upward instead of widgets reaching across domains.
- Treat async workers, timers, file watchers, and service calls as owned resources with explicit stop/shutdown paths.
- Test DOM/state contracts, focus behavior, density, and route changes with Textual pilot tests.

## File Structure

### Create

- `tldw_chatbook/UI/Workbench/__init__.py`
  - Public exports for workbench state, widgets, focus/help helpers, and route inventory helpers.
- `tldw_chatbook/UI/Workbench/workbench_state.py`
  - Immutable state dataclasses for headers, modes, actions, panes, density, recovery, and status.
- `tldw_chatbook/UI/Workbench/workbench_widgets.py`
  - Textual widgets: `WorkbenchFrame`, `DestinationHeader`, `ModeStrip`, `CommandStrip`, `WorkbenchPane`, `RecoveryCallout`, `StateBlock`.
- `tldw_chatbook/UI/Workbench/focus.py`
  - Pure focus-order model plus small helpers for F6 pane cycling.
- `tldw_chatbook/UI/Workbench/help.py`
  - Contextual help state and a help panel/modal that exposes visible actions for the current workbench.
- `tldw_chatbook/UI/Workbench/route_inventory.py`
  - Canonical route coverage model and migration-owner map for all current routes and aliases.
- `tldw_chatbook/Utils/ui_responsiveness.py`
  - Lightweight event-loop heartbeat, worker/timer counters, mount-churn counters, and snapshot formatter.
- `tldw_chatbook/Utils/ui_responsiveness_artifacts.py`
  - Writes required diagnostics artifacts: `ui_heartbeat.log`, `worker_snapshot.log`, `timer_registry.log`, `mount_churn_summary.log`, and `route_switch_soak_result.txt`.
- `tldw_chatbook/Widgets/Console/console_workbench_state.py`
  - Adapter from Console display/readiness/session state to shared `WorkbenchState`.
- `Tests/UI/run_workbench_soak.py`
  - Runnable Textual pilot harness for route switching, Console idle, and synthetic streaming/stop cycles.
- `tldw_chatbook/css/components/_workbench.tcss`
  - Shared workbench TCSS classes and density contracts.
- `Tests/UI/test_workbench_route_inventory.py`
  - Route inventory and migration-owner coverage tests.
- `Tests/UI/test_ui_responsiveness.py`
  - Unit tests for heartbeat, counters, threshold snapshots, and app-facing formatter.
- `Tests/UI/test_ui_responsiveness_artifacts.py`
  - Unit tests that artifact writer and soak summary include all required filenames and thresholds.
- `Tests/UI/test_workbench_state.py`
  - Unit tests for state validation and class derivation.
- `Tests/UI/test_workbench_widgets.py`
  - Textual pilot tests for stable sync, visible actions, recovery state, and density classes.
- `Tests/UI/test_workbench_focus_help.py`
  - Tests for F6 order, contextual help contents, and footer shortcut context.
- `Tests/UI/test_workbench_visual_snapshots.py`
  - `export_screenshot` gates for normal density, compact density, command palette, and focus states.
- `Tests/UI/test_console_workbench_contract.py`
  - Console reference tests for visible controls, provider/model recovery, focus cycling, and no command-palette dependency.
- `Tests/UI/test_console_workbench_parity_matrix.py`
  - Tracks every Console reference parity requirement and the test command that proves it.
- `Docs/Design/chatbook-workbench-ui-system.md`
  - Durable design-system notes for future destination migrations.
- `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`
  - Manual/automated verification evidence, including responsiveness baseline commands.

### Modify

- `tldw_chatbook/UI/Navigation/screen_registry.py`
  - Add safe read-only route registry helpers so tests do not rely on private globals.
- `tldw_chatbook/UI/Navigation/shell_destinations.py`
  - Add safe read-only shell route helpers for migration coverage tests.
- `tldw_chatbook/app.py:1126-1139`
  - Add visible global bindings for F1 contextual help and F6 focus next pane.
- `tldw_chatbook/app.py:1193-1197`
  - Store the `UIResponsivenessMonitor`.
- `tldw_chatbook/app.py:3155-3189`
  - Start the responsiveness monitor with the main UI and expose footer context.
- `tldw_chatbook/app.py:4549-4616`
  - Record deferred timers/status-update timers in the responsiveness monitor.
- `tldw_chatbook/Widgets/AppFooterStatus.py`
  - Add workbench-aware footer context rendering without removing existing word/token/DB status behavior.
- `tldw_chatbook/UI/Screens/chat_screen.py:2582-2916`
  - Replace local hidden/static/frame helper code with shared workbench state/widgets where possible.
- `tldw_chatbook/UI/Screens/chat_screen.py:3085-3368`
  - Refactor Console layout to use `WorkbenchFrame`, `DestinationHeader`, `ModeStrip`, `CommandStrip`, `WorkbenchPane`, and `RecoveryCallout` while preserving the left rail/main/right rail/composer framing.
- `tldw_chatbook/UI/Screens/chat_screen.py:3394-3425`
  - Register Console focus/help context and stop Console-owned timers/monitor entries on unmount.
- `tldw_chatbook/Widgets/Console/console_control_bar.py`
  - Convert hidden label seams into visible mode/action summaries or retire the hidden widget after parity tests pass.
- `tldw_chatbook/Widgets/Console/console_session_surface.py`
  - Keep stable session tab reconciliation, but expose state through workbench pane classes.
- `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  - Ensure send/stop/attach/save actions are visible and surfaced in workbench help/footer state.
- `tldw_chatbook/Widgets/Console/__init__.py`
  - Export `console_workbench_state` helpers if needed by tests.
- `tldw_chatbook/css/main.tcss`
  - Import `components/_workbench.tcss` after `_agentic_terminal.tcss`.
- `tldw_chatbook/css/build_css.py`
  - Add `components/_workbench.tcss` to `CSS_MODULES`.
- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Regenerate with `python3 tldw_chatbook/css/build_css.py`.
- `Docs/Design/master-shell-design-system-contract.md`
  - Link to the Workbench UI System doc and note the visible-action rule.
- `backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md`
  - Keep `Implementation Plan`, acceptance criteria, and final notes aligned with this plan.

## Migration Owner Map

Use this map in `route_inventory.py`. Every current route or alias must either have an owner here or fail `Tests/UI/test_workbench_route_inventory.py`.
Owner IDs normalize the spec's migration-owner labels. For example, `artifacts_writing` represents `Artifacts / Writing`, and `diagnostics_evals`, `diagnostics_stats`, and `diagnostics_logs` represent the Diagnostics owners from the approved spec.

```python
WORKBENCH_ROUTE_OWNERS: dict[str, str] = {
    "home": "home",
    "chat": "console",
    "console": "console",
    "library": "library",
    "artifacts": "artifacts",
    "personas": "personas",
    "watchlists_collections": "watchlists_collections",
    "schedules": "schedules",
    "workflows": "workflows",
    "mcp": "mcp",
    "acp": "acp",
    "skills": "skills",
    "settings": "settings",
    "ingest": "library",
    "coding": "console",
    "conversation": "library",
    "ccp": "personas",
    "conversations_characters_prompts": "personas",
    "characters": "personas",
    "prompts": "personas",
    "media": "library",
    "notes": "library",
    "search": "library",
    "evals": "diagnostics_evals",
    "tools_settings": "mcp",
    "llm": "settings",
    "llm_management": "settings",
    "customize": "settings",
    "logs": "diagnostics_logs",
    "stats": "diagnostics_stats",
    "stts": "settings",
    "study": "library",
    "writing": "artifacts_writing",
    "research": "library",
    "chatbooks": "artifacts",
    "subscriptions": "watchlists_collections",
    "subscription": "watchlists_collections",
}
```

## ASCII Target Frame

```text
+----------------------------------------------------------------------------+
| Console                    Mode: Chat/RAG/Follow       Provider: ready      |
| [New tab] [Settings] [Attach] [Run Library RAG] [Save] [Help]               |
+-------------+------------------------------------------+-------------------+
| Context     | Transcript / Event Stream                | Inspector         |
| Workspace   |                                          | Provider/model    |
| Staged      | Empty/loading/error/recovery/messages    | Tools/approvals   |
| Sources     |                                          | Run evidence      |
+-------------+------------------------------------------+-------------------+
| Composer: ask, command, paste task...                   [Send] [Stop]       |
+----------------------------------------------------------------------------+
| Footer: F6 next pane | F1 help | Ctrl+P palette | route-specific hints      |
```

---

### Task 1: Route Inventory And Planning Hygiene

**Files:**
- Create: `tldw_chatbook/UI/Workbench/__init__.py`
- Create: `tldw_chatbook/UI/Workbench/route_inventory.py`
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py:36-100`
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py:88-154`
- Test: `Tests/UI/test_workbench_route_inventory.py`
- Modify: `backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md`

- [ ] **Step 1: Write the failing route inventory tests**

```python
from tldw_chatbook.UI.Workbench.route_inventory import (
    WORKBENCH_ROUTE_OWNERS,
    build_workbench_route_coverage,
)


def test_all_registered_screen_routes_have_workbench_migration_owner():
    coverage = build_workbench_route_coverage()

    assert coverage.missing_owner_routes == ()
    assert "chat" in coverage.screen_routes
    assert WORKBENCH_ROUTE_OWNERS["chat"] == "console"


def test_shell_legacy_aliases_have_workbench_migration_owner():
    coverage = build_workbench_route_coverage()

    for alias in ("conversations_characters_prompts", "characters", "prompts", "subscription"):
        assert alias in coverage.all_known_routes
        assert alias not in coverage.missing_owner_routes


def test_route_coverage_exposes_future_destination_owners():
    coverage = build_workbench_route_coverage()

    assert coverage.owner_for_route["notes"] == "library"
    assert coverage.owner_for_route["console"] == "console"
    assert coverage.owner_for_route["tools_settings"] == "mcp"
    assert coverage.owner_for_route["llm_management"] == "settings"
    assert coverage.owner_for_route["writing"] == "artifacts_writing"
    assert coverage.owner_for_route["evals"] == "diagnostics_evals"
    assert coverage.owner_for_route["stats"] == "diagnostics_stats"
    assert coverage.owner_for_route["logs"] == "diagnostics_logs"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_workbench_route_inventory.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.UI.Workbench'`.

- [ ] **Step 3: Add read-only registry helpers**

In `tldw_chatbook/UI/Navigation/screen_registry.py`, add:

```python
def registered_screen_route_ids() -> tuple[str, ...]:
    """Return all registered screen route ids without loading screen classes."""

    return tuple(sorted(_SCREEN_ROUTES))


def registered_screen_aliases() -> tuple[str, ...]:
    """Return screen route aliases without loading screen classes."""

    return tuple(sorted(_SCREEN_ALIASES))
```

In `tldw_chatbook/UI/Navigation/shell_destinations.py`, add:

```python
def registered_shell_route_ids() -> tuple[str, ...]:
    """Return all shell route ids and aliases known to the destination model."""

    return tuple(sorted(_ROUTE_MAP))
```

- [ ] **Step 4: Implement route inventory module**

Create `tldw_chatbook/UI/Workbench/__init__.py`:

```python
"""Shared Textual workbench primitives for Chatbook destinations."""
```

Create `tldw_chatbook/UI/Workbench/route_inventory.py`:

```python
"""Route coverage helpers for the Workbench UI migration."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_chatbook.Constants import ALL_TABS
from tldw_chatbook.UI.Navigation.screen_registry import (
    registered_screen_aliases,
    registered_screen_route_ids,
)
from tldw_chatbook.UI.Navigation.shell_destinations import registered_shell_route_ids


WORKBENCH_ROUTE_OWNERS: dict[str, str] = {
    "home": "home",
    "chat": "console",
    "console": "console",
    "library": "library",
    "artifacts": "artifacts",
    "personas": "personas",
    "watchlists_collections": "watchlists_collections",
    "schedules": "schedules",
    "workflows": "workflows",
    "mcp": "mcp",
    "acp": "acp",
    "skills": "skills",
    "settings": "settings",
    "ingest": "library",
    "coding": "console",
    "conversation": "library",
    "ccp": "personas",
    "conversations_characters_prompts": "personas",
    "characters": "personas",
    "prompts": "personas",
    "media": "library",
    "notes": "library",
    "search": "library",
    "evals": "diagnostics_evals",
    "tools_settings": "mcp",
    "llm": "settings",
    "llm_management": "settings",
    "customize": "settings",
    "logs": "diagnostics_logs",
    "stats": "diagnostics_stats",
    "stts": "settings",
    "study": "library",
    "writing": "artifacts_writing",
    "research": "library",
    "chatbooks": "artifacts",
    "subscriptions": "watchlists_collections",
    "subscription": "watchlists_collections",
}


@dataclass(frozen=True)
class WorkbenchRouteCoverage:
    constant_tabs: tuple[str, ...]
    screen_routes: tuple[str, ...]
    screen_aliases: tuple[str, ...]
    shell_routes: tuple[str, ...]
    all_known_routes: tuple[str, ...]
    owner_for_route: dict[str, str]
    missing_owner_routes: tuple[str, ...]


def build_workbench_route_coverage() -> WorkbenchRouteCoverage:
    """Return route coverage for all registered navigation sources."""

    constant_tabs = tuple(sorted(str(route) for route in ALL_TABS))
    screen_routes = registered_screen_route_ids()
    screen_aliases = registered_screen_aliases()
    shell_routes = registered_shell_route_ids()
    all_known_routes = tuple(
        sorted(set(constant_tabs) | set(screen_routes) | set(screen_aliases) | set(shell_routes))
    )
    missing_owner_routes = tuple(
        route for route in all_known_routes if route not in WORKBENCH_ROUTE_OWNERS
    )
    return WorkbenchRouteCoverage(
        constant_tabs=constant_tabs,
        screen_routes=screen_routes,
        screen_aliases=screen_aliases,
        shell_routes=shell_routes,
        all_known_routes=all_known_routes,
        owner_for_route={route: WORKBENCH_ROUTE_OWNERS[route] for route in all_known_routes if route in WORKBENCH_ROUTE_OWNERS},
        missing_owner_routes=missing_owner_routes,
    )
```

- [ ] **Step 5: Run route inventory tests**

Run: `pytest Tests/UI/test_workbench_route_inventory.py -q`

Expected: PASS.

- [ ] **Step 6: Link the plan in the Backlog task**

Add an `## Implementation Plan` section to `backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md` with:

```markdown
ADR required: yes
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: shared UI architecture, route ownership, and responsiveness gates.

Plan: Docs/superpowers/plans/2026-06-29-chatbook-workbench-ui-foundation-console-plan.md
```

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Navigation/screen_registry.py \
  tldw_chatbook/UI/Navigation/shell_destinations.py \
  tldw_chatbook/UI/Workbench/__init__.py \
  tldw_chatbook/UI/Workbench/route_inventory.py \
  Tests/UI/test_workbench_route_inventory.py \
  "backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md"
git commit -m "Add workbench route migration coverage"
```

---

### Task 2: Responsiveness Instrumentation Baseline

**Files:**
- Create: `tldw_chatbook/Utils/ui_responsiveness.py`
- Create: `tldw_chatbook/Utils/ui_responsiveness_artifacts.py`
- Create: `Tests/UI/run_workbench_soak.py`
- Modify: `tldw_chatbook/app.py:1120-1197`
- Modify: `tldw_chatbook/app.py:3155-3189`
- Modify: `tldw_chatbook/app.py:4549-4616`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:379-390`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:4275`
- Test: `Tests/UI/test_ui_responsiveness.py`
- Test: `Tests/UI/test_ui_responsiveness_artifacts.py`
- QA: `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`

- [ ] **Step 1: Write failing unit tests for monitor counters**

```python
from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor


def test_responsiveness_snapshot_records_timers_workers_and_mount_churn():
    monitor = UIResponsivenessMonitor(enabled=True, stall_threshold_ms=100)

    monitor.record_timer_created("footer-token")
    monitor.record_worker_started("console-sync")
    monitor.record_mounts("console-tabs", mounted=3, removed=1)
    monitor.record_worker_finished("console-sync")

    snapshot = monitor.snapshot()

    assert snapshot.active_timers == 1
    assert snapshot.active_workers == 0
    assert snapshot.mounts == 3
    assert snapshot.removes == 1
    assert "timers=1" in snapshot.format_status_line()


def test_responsiveness_snapshot_marks_event_loop_stall():
    monitor = UIResponsivenessMonitor(enabled=True, stall_threshold_ms=50)

    monitor.record_heartbeat_delta(0.075)

    snapshot = monitor.snapshot()

    assert snapshot.max_heartbeat_lag_ms == 75
    assert snapshot.stalled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_ui_responsiveness.py -q`

Expected: FAIL with `ModuleNotFoundError` or missing `UIResponsivenessMonitor`.

- [ ] **Step 3: Implement the monitor**

Create `tldw_chatbook/Utils/ui_responsiveness.py` with deterministic methods first. Keep Textual integration thin.

```python
"""Lightweight UI responsiveness instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class UIResponsivenessSnapshot:
    enabled: bool
    active_timers: int
    active_workers: int
    mounts: int
    removes: int
    max_heartbeat_lag_ms: int
    stalled: bool

    def format_status_line(self) -> str:
        if not self.enabled:
            return "UI diag: disabled"
        state = "stalled" if self.stalled else "responsive"
        return (
            f"UI diag: {state} | lag={self.max_heartbeat_lag_ms}ms | "
            f"workers={self.active_workers} | timers={self.active_timers} | "
            f"mounts={self.mounts} removes={self.removes}"
        )


class UIResponsivenessMonitor:
    """Collect low-cost counters that make UI stalls diagnosable."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        stall_threshold_ms: int = 250,
        heartbeat_interval_seconds: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.stall_threshold_ms = stall_threshold_ms
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._active_timers: set[str] = set()
        self._active_workers: set[str] = set()
        self._mounts = 0
        self._removes = 0
        self._max_heartbeat_lag_ms = 0
        self._last_heartbeat = time.perf_counter()

    def record_timer_created(self, name: str) -> None:
        if self.enabled:
            self._active_timers.add(name)

    def record_timer_stopped(self, name: str) -> None:
        self._active_timers.discard(name)

    def record_worker_started(self, name: str) -> None:
        if self.enabled:
            self._active_workers.add(name)

    def record_worker_finished(self, name: str) -> None:
        self._active_workers.discard(name)

    def record_mounts(self, owner: str, *, mounted: int = 0, removed: int = 0) -> None:
        if not self.enabled:
            return
        self._mounts += max(0, mounted)
        self._removes += max(0, removed)

    def record_heartbeat_delta(self, delta_seconds: float) -> None:
        """Record event-loop drift beyond the configured heartbeat cadence."""

        if not self.enabled:
            return
        self._max_heartbeat_lag_ms = max(
            self._max_heartbeat_lag_ms,
            int(round(delta_seconds * 1000)),
        )

    def heartbeat(self) -> None:
        now = time.perf_counter()
        elapsed_seconds = now - self._last_heartbeat
        lag_seconds = max(0.0, elapsed_seconds - self.heartbeat_interval_seconds)
        self.record_heartbeat_delta(lag_seconds)
        self._last_heartbeat = now

    def snapshot(self) -> UIResponsivenessSnapshot:
        return UIResponsivenessSnapshot(
            enabled=self.enabled,
            active_timers=len(self._active_timers),
            active_workers=len(self._active_workers),
            mounts=self._mounts,
            removes=self._removes,
            max_heartbeat_lag_ms=self._max_heartbeat_lag_ms,
            stalled=self._max_heartbeat_lag_ms >= self.stall_threshold_ms,
        )
```

- [ ] **Step 4: Run monitor unit tests**

Run: `pytest Tests/UI/test_ui_responsiveness.py -q`

Expected: PASS.

- [ ] **Step 5: Add app-level monitor wiring**

Modify `tldw_chatbook/app.py`:

- Import `UIResponsivenessMonitor`.
- Add `ui_responsiveness_monitor: UIResponsivenessMonitor | None = None` near the footer timer fields.
- Initialize it in `__init__` or before main UI composition using a config flag such as `diagnostics.ui_responsiveness_enabled`, defaulting to `True`.
- In `_create_main_ui_widgets`, call `self._start_ui_responsiveness_monitor()` before returning widgets.
- Add:

```python
def _start_ui_responsiveness_monitor(self) -> None:
    interval_seconds = 1.0
    if self.ui_responsiveness_monitor is None:
        enabled = bool(get_cli_setting("diagnostics", "ui_responsiveness_enabled", True))
        self.ui_responsiveness_monitor = UIResponsivenessMonitor(
            enabled=enabled,
            heartbeat_interval_seconds=interval_seconds,
        )
    self.ui_responsiveness_monitor.record_timer_created("ui-heartbeat")
    self.set_interval(interval_seconds, self._record_ui_heartbeat)


def _record_ui_heartbeat(self) -> None:
    monitor = self.ui_responsiveness_monitor
    if monitor is not None:
        monitor.heartbeat()
```

When adding timers in `_schedule_footer_status_updates`, call `record_timer_created()` with stable names such as `footer-db-size-once`, `footer-db-size-periodic`, and `footer-token-periodic`.

- [ ] **Step 6: Instrument Console high-churn paths**

In `tldw_chatbook/UI/Screens/chat_screen.py`:

- Around `self.run_worker(self._sync_native_console_chat_ui(), exclusive=True)`, record worker start/finish where the worker has a stable name.
- In `ConsoleSessionSurface.sync_sessions`, record tab strip removes/mounts through `app_instance.ui_responsiveness_monitor` after the reconciliation branch.
- When `_console_transcript_sync_timer` is created or stopped, record `console-transcript-sync`.

Keep this instrumentation best-effort: it must never raise if the monitor is absent.

- [ ] **Step 7: Add app-facing tests**

Extend `Tests/UI/test_ui_responsiveness.py`:

```python
def test_responsiveness_status_line_is_footer_safe():
    monitor = UIResponsivenessMonitor(enabled=True, stall_threshold_ms=50)
    monitor.record_timer_created("ui-heartbeat")
    monitor.record_heartbeat_delta(0.01)

    line = monitor.snapshot().format_status_line()

    assert "\n" not in line
    assert "UI diag: responsive" in line
```

Run: `pytest Tests/UI/test_ui_responsiveness.py -q`

Expected: PASS.

- [ ] **Step 8: Write failing artifact tests**

Create `Tests/UI/test_ui_responsiveness_artifacts.py`:

```python
from pathlib import Path

from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor
from tldw_chatbook.Utils.ui_responsiveness_artifacts import (
    REQUIRED_RESPONSIVENESS_ARTIFACTS,
    write_responsiveness_artifacts,
)


def test_responsiveness_artifact_writer_creates_required_files(tmp_path: Path):
    monitor = UIResponsivenessMonitor(enabled=True, stall_threshold_ms=250)
    monitor.record_timer_created("ui-heartbeat")
    monitor.record_worker_started("console-sync")
    monitor.record_mounts("console-tabs", mounted=2, removed=1)
    monitor.record_heartbeat_delta(0.03)

    write_responsiveness_artifacts(
        tmp_path,
        monitor.snapshot(),
        route_switch_summary="route switches: 6, failures: 0",
    )

    for filename in REQUIRED_RESPONSIVENESS_ARTIFACTS:
        assert (tmp_path / filename).exists(), filename
    assert "route switches: 6" in (tmp_path / "route_switch_soak_result.txt").read_text(encoding="utf-8")
```

- [ ] **Step 9: Run artifact tests to verify they fail**

Run: `pytest Tests/UI/test_ui_responsiveness_artifacts.py -q`

Expected: FAIL with missing `ui_responsiveness_artifacts`.

- [ ] **Step 10: Implement artifact writer**

Create `tldw_chatbook/Utils/ui_responsiveness_artifacts.py`:

```python
"""Write Workbench responsiveness diagnostics artifacts."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessSnapshot


REQUIRED_RESPONSIVENESS_ARTIFACTS = (
    "ui_heartbeat.log",
    "worker_snapshot.log",
    "timer_registry.log",
    "mount_churn_summary.log",
    "route_switch_soak_result.txt",
)


def write_responsiveness_artifacts(
    output_dir: Path,
    snapshot: UIResponsivenessSnapshot,
    *,
    route_switch_summary: str,
) -> None:
    """Write the required freeze-diagnostics artifacts for a soak run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ui_heartbeat.log").write_text(
        f"max_heartbeat_lag_ms={snapshot.max_heartbeat_lag_ms}\nstalled={snapshot.stalled}\n",
        encoding="utf-8",
    )
    (output_dir / "worker_snapshot.log").write_text(
        f"active_workers={snapshot.active_workers}\n",
        encoding="utf-8",
    )
    (output_dir / "timer_registry.log").write_text(
        f"active_timers={snapshot.active_timers}\n",
        encoding="utf-8",
    )
    (output_dir / "mount_churn_summary.log").write_text(
        f"mounts={snapshot.mounts}\nremoves={snapshot.removes}\n",
        encoding="utf-8",
    )
    (output_dir / "route_switch_soak_result.txt").write_text(
        f"{route_switch_summary}\n",
        encoding="utf-8",
    )
```

- [ ] **Step 11: Add runnable soak harness**

Create `Tests/UI/run_workbench_soak.py` as a small async script that:

- Builds the test app with `_build_test_app(initial_tab="chat")`.
- Opens Console and waits for `#console-shell`.
- Switches among `chat`, `library`, `settings`, and back to `chat` at least six times.
- Returns to Console after route switches and verifies focus is restored to a sensible Workbench target such as the composer or transcript surface.
- Toggles Console rails if their buttons are present.
- Starts/stops a synthetic transcript/composer interaction without requiring a real provider.
- Writes artifacts to `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/`.
- Exits non-zero if route switches fail, if focus restoration fails, if `snapshot.stalled` is true after the drift threshold, if active workers grow after the run, or if any required artifact is missing.

Command:

```bash
python3 Tests/UI/run_workbench_soak.py --output Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts --route-switches 6 --idle-seconds 10
```

Expected: creates:

```text
Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/ui_heartbeat.log
Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/worker_snapshot.log
Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/timer_registry.log
Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/mount_churn_summary.log
Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/route_switch_soak_result.txt
```

- [ ] **Step 12: Run responsiveness tests**

Run:

```bash
pytest Tests/UI/test_ui_responsiveness.py Tests/UI/test_ui_responsiveness_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 13: Capture baseline evidence**

Append to `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`:

```markdown
## Responsiveness Baseline

- Command: `pytest Tests/UI/test_ui_responsiveness.py -q`
- Evidence: monitor records heartbeat lag, active timers, active workers, mount count, and remove count.
- Command: `pytest Tests/UI/test_ui_responsiveness_artifacts.py -q`
- Evidence: artifact writer creates every required freeze-diagnostics artifact.
- Command: `python3 Tests/UI/run_workbench_soak.py --output Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts --route-switches 6 --idle-seconds 10`
- Evidence: route switch soak creates `ui_heartbeat.log`, `worker_snapshot.log`, `timer_registry.log`, `mount_churn_summary.log`, and `route_switch_soak_result.txt`.
```

- [ ] **Step 14: Commit**

```bash
git add tldw_chatbook/Utils/ui_responsiveness.py \
  tldw_chatbook/Utils/ui_responsiveness_artifacts.py \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/Widgets/Console/console_session_surface.py \
  Tests/UI/test_ui_responsiveness.py \
  Tests/UI/test_ui_responsiveness_artifacts.py \
  Tests/UI/run_workbench_soak.py \
  Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md
git commit -m "Add UI responsiveness instrumentation baseline"
```

---

### Task 3: Shared Workbench State And Widgets

**Files:**
- Create: `tldw_chatbook/UI/Workbench/workbench_state.py`
- Create: `tldw_chatbook/UI/Workbench/workbench_widgets.py`
- Modify: `tldw_chatbook/UI/Workbench/__init__.py`
- Create: `tldw_chatbook/css/components/_workbench.tcss`
- Modify: `tldw_chatbook/css/main.tcss:31-39`
- Modify: `tldw_chatbook/css/build_css.py:26-37`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_workbench_state.py`
- Test: `Tests/UI/test_workbench_widgets.py`

- [ ] **Step 1: Write failing state tests**

```python
import pytest

from tldw_chatbook.UI.Workbench.workbench_state import (
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchState,
)


def test_workbench_state_rejects_duplicate_action_ids():
    with pytest.raises(ValueError, match="duplicate action id"):
        WorkbenchState(
            header=WorkbenchHeaderState(title="Console"),
            actions=(
                WorkbenchAction("settings", "Settings"),
                WorkbenchAction("settings", "Settings again"),
            ),
        )


def test_workbench_mode_classes_include_status_and_density():
    mode = WorkbenchMode(id="rag", label="RAG", active=True, status="ready")

    assert mode.css_classes == "workbench-mode is-active status-ready"
```

- [ ] **Step 2: Run state tests to verify they fail**

Run: `pytest Tests/UI/test_workbench_state.py -q`

Expected: FAIL with missing module/classes.

- [ ] **Step 3: Implement state dataclasses**

Create `tldw_chatbook/UI/Workbench/workbench_state.py`:

```python
"""State snapshots for shared Workbench UI widgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Density = Literal["normal", "compact"]
WorkbenchStatus = Literal["ready", "running", "blocked", "error", "paused", "empty", "loading"]


@dataclass(frozen=True)
class WorkbenchAction:
    id: str
    label: str
    tooltip: str = ""
    disabled: bool = False
    primary: bool = False

    @property
    def css_classes(self) -> str:
        classes = ["workbench-action"]
        if self.primary:
            classes.append("is-primary")
        if self.disabled:
            classes.append("is-disabled")
        return " ".join(classes)


@dataclass(frozen=True)
class WorkbenchMode:
    id: str
    label: str
    active: bool = False
    status: WorkbenchStatus = "ready"

    @property
    def css_classes(self) -> str:
        classes = ["workbench-mode"]
        if self.active:
            classes.append("is-active")
        classes.append(f"status-{self.status}")
        return " ".join(classes)


@dataclass(frozen=True)
class WorkbenchHeaderState:
    title: str
    subtitle: str = ""
    status: WorkbenchStatus = "ready"
    density: Density = "normal"


@dataclass(frozen=True)
class WorkbenchPaneState:
    id: str
    title: str
    status: WorkbenchStatus = "ready"
    collapsed: bool = False


@dataclass(frozen=True)
class RecoveryState:
    title: str
    body: str
    action: WorkbenchAction | None = None
    visible: bool = True


@dataclass(frozen=True)
class WorkbenchState:
    header: WorkbenchHeaderState
    modes: tuple[WorkbenchMode, ...] = ()
    actions: tuple[WorkbenchAction, ...] = ()
    panes: tuple[WorkbenchPaneState, ...] = ()
    recovery: RecoveryState | None = None
    density: Density = "normal"
    route_id: str = ""

    def __post_init__(self) -> None:
        action_ids = [action.id for action in self.actions]
        duplicates = sorted({action_id for action_id in action_ids if action_ids.count(action_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate action id: {', '.join(duplicates)}")
```

- [ ] **Step 4: Run state tests**

Run: `pytest Tests/UI/test_workbench_state.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing widget sync tests**

```python
import pytest
from textual.app import App

from tldw_chatbook.UI.Workbench.workbench_state import (
    RecoveryState,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchState,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested, WorkbenchFrame


class WorkbenchFrameApp(App):
    def __init__(self, state: WorkbenchState):
        super().__init__()
        self.state = state
        self.received_actions: list[str] = []

    def compose(self):
        yield WorkbenchFrame(self.state, id="frame")

    def on_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
        self.received_actions.append(event.action_id)


@pytest.mark.asyncio
async def test_workbench_frame_sync_keeps_child_ids_stable():
    state = WorkbenchState(
        header=WorkbenchHeaderState(title="Console"),
        actions=(WorkbenchAction("settings", "Settings"),),
    )
    app = WorkbenchFrameApp(state)

    async with app.run_test(size=(100, 24)) as pilot:
        frame = app.query_one("#frame", WorkbenchFrame)
        before = [child.id for child in frame.children]
        frame.sync_state(
            WorkbenchState(
                header=WorkbenchHeaderState(title="Console", subtitle="Ready"),
                actions=(WorkbenchAction("settings", "Settings", primary=True),),
                recovery=RecoveryState("Provider setup needed", "Choose a model"),
            )
        )
        await pilot.pause()

        assert [child.id for child in frame.children] == before
        assert "Choose a model" in frame.query_one("#workbench-recovery").renderable.plain


@pytest.mark.asyncio
async def test_recovery_callout_renders_and_emits_action():
    state = WorkbenchState(
        header=WorkbenchHeaderState(title="Console"),
        recovery=RecoveryState(
            "Provider setup needed",
            "Choose a model",
            action=WorkbenchAction("provider-recovery", "Choose model", primary=True),
        ),
    )
    app = WorkbenchFrameApp(state)

    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        action = app.query_one("#workbench-recovery-action")
        assert action.label == "Choose model"
        assert not action.disabled
        await pilot.click("#workbench-recovery-action")

    assert app.received_actions == ["provider-recovery"]
```

- [ ] **Step 6: Run widget tests to verify they fail**

Run: `pytest Tests/UI/test_workbench_widgets.py -q`

Expected: FAIL with missing widgets.

- [ ] **Step 7: Implement workbench widgets**

Create `tldw_chatbook/UI/Workbench/workbench_widgets.py` using stable children and `sync_state()` methods. Emit typed messages for action buttons.

```python
"""Shared Textual widgets for destination workbench screens."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.UI.Workbench.workbench_state import (
    RecoveryState,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchPaneState,
    WorkbenchState,
)


class WorkbenchActionRequested(Message):
    def __init__(self, action_id: str) -> None:
        self.action_id = action_id
        super().__init__()


class DestinationHeader(Vertical):
    def __init__(self, state: WorkbenchHeaderState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(self.state.title, id="workbench-header-title", classes="workbench-header-title")
        yield Static(self.state.subtitle, id="workbench-header-subtitle", classes="workbench-header-subtitle")

    def sync_state(self, state: WorkbenchHeaderState) -> None:
        self.state = state
        self.set_class(state.density == "compact", "density-compact")
        self.set_class(state.density == "normal", "density-normal")
        self.query_one("#workbench-header-title", Static).update(state.title)
        self.query_one("#workbench-header-subtitle", Static).update(state.subtitle)


class CommandStrip(Horizontal):
    def __init__(self, actions: tuple[WorkbenchAction, ...], **kwargs):
        super().__init__(**kwargs)
        self.actions = actions

    def compose(self) -> ComposeResult:
        for action in self.actions:
            yield self._build_button(action)

    def _build_button(self, action: WorkbenchAction) -> Button:
        button = Button(action.label, id=f"workbench-action-{action.id}", classes=action.css_classes, compact=True)
        button.tooltip = action.tooltip or action.label
        button.disabled = action.disabled
        return button

    def sync_actions(self, actions: tuple[WorkbenchAction, ...]) -> None:
        self.actions = actions
        for action in actions:
            selector = f"#workbench-action-{action.id}"
            matches = list(self.query(selector))
            if matches and isinstance(matches[0], Button):
                button = matches[0]
                button.label = action.label
                button.tooltip = action.tooltip or action.label
                button.disabled = action.disabled
                button.set_class(action.primary, "is-primary")
                button.set_class(action.disabled, "is-disabled")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("workbench-action-"):
            return
        event.stop()
        self.post_message(WorkbenchActionRequested(button_id.removeprefix("workbench-action-")))


class ModeStrip(Horizontal):
    def __init__(self, modes: tuple[WorkbenchMode, ...], **kwargs):
        super().__init__(**kwargs)
        self.modes = modes

    def compose(self) -> ComposeResult:
        for mode in self.modes:
            yield Static(mode.label, id=f"workbench-mode-{mode.id}", classes=mode.css_classes)

    def sync_modes(self, modes: tuple[WorkbenchMode, ...]) -> None:
        self.modes = modes
        for mode in modes:
            matches = list(self.query(f"#workbench-mode-{mode.id}"))
            if not matches:
                continue
            widget = matches[0]
            if isinstance(widget, Static):
                widget.update(mode.label)
                widget.set_class(mode.active, "is-active")
                for status in ("ready", "running", "blocked", "error", "paused", "empty", "loading"):
                    widget.set_class(mode.status == status, f"status-{status}")


class RecoveryCallout(Vertical):
    def __init__(self, recovery: RecoveryState | None, **kwargs):
        super().__init__(**kwargs)
        self.recovery = recovery

    def compose(self) -> ComposeResult:
        yield Static("", id="workbench-recovery-title")
        yield Static("", id="workbench-recovery")
        action_button = Button(
            "Recover",
            id="workbench-recovery-action",
            classes="workbench-recovery-action",
            compact=True,
        )
        action_button.styles.display = "none"
        yield action_button

    def on_mount(self) -> None:
        self.sync_recovery(self.recovery)

    def sync_recovery(self, recovery: RecoveryState | None) -> None:
        self.recovery = recovery
        visible = recovery is not None and recovery.visible
        self.styles.display = "block" if visible else "none"
        self.query_one("#workbench-recovery-title", Static).update(recovery.title if visible else "")
        self.query_one("#workbench-recovery", Static).update(recovery.body if visible else "")
        action_button = self.query_one("#workbench-recovery-action", Button)
        action_visible = visible and recovery.action is not None
        action_button.styles.display = "block" if action_visible else "none"
        action_button.disabled = not action_visible or bool(recovery.action and recovery.action.disabled)
        if recovery and recovery.action:
            action_button.label = recovery.action.label
            action_button.tooltip = recovery.action.tooltip or recovery.action.label

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "workbench-recovery-action" or self.recovery is None or self.recovery.action is None:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested(self.recovery.action.id))


class WorkbenchPane(Vertical):
    def __init__(self, state: WorkbenchPaneState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(self.state.title, id=f"workbench-pane-title-{self.state.id}", classes="workbench-pane-title")

    def sync_state(self, state: WorkbenchPaneState) -> None:
        self.state = state
        self.set_class(state.collapsed, "is-collapsed")
        for status in ("ready", "running", "blocked", "error", "paused", "empty", "loading"):
            self.set_class(state.status == status, f"status-{status}")
        self.query_one(f"#workbench-pane-title-{state.id}", Static).update(state.title)


class StateBlock(Vertical):
    def __init__(self, title: str = "", body: str = "", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.body = body

    def compose(self) -> ComposeResult:
        yield Static(self.title, id="workbench-state-title", classes="workbench-state-title")
        yield Static(self.body, id="workbench-state-body", classes="workbench-state-body")

    def sync_copy(self, *, title: str, body: str) -> None:
        self.title = title
        self.body = body
        self.query_one("#workbench-state-title", Static).update(title)
        self.query_one("#workbench-state-body", Static).update(body)


class WorkbenchFrame(Vertical):
    def __init__(self, state: WorkbenchState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield DestinationHeader(self.state.header, id="workbench-header", classes="workbench-header")
        yield ModeStrip(self.state.modes, id="workbench-mode-strip", classes="workbench-mode-strip")
        yield CommandStrip(self.state.actions, id="workbench-command-strip", classes="workbench-command-strip")
        yield RecoveryCallout(self.state.recovery, id="workbench-recovery-callout", classes="workbench-recovery-callout")

    def sync_state(self, state: WorkbenchState) -> None:
        self.state = state
        self.set_class(state.density == "compact", "density-compact")
        self.set_class(state.density == "normal", "density-normal")
        self.query_one("#workbench-header", DestinationHeader).sync_state(state.header)
        self.query_one("#workbench-mode-strip", ModeStrip).sync_modes(state.modes)
        self.query_one("#workbench-command-strip", CommandStrip).sync_actions(state.actions)
        self.query_one("#workbench-recovery-callout", RecoveryCallout).sync_recovery(state.recovery)
```

Before proceeding to Console, confirm `ModeStrip`, `WorkbenchPane`, and `StateBlock` have stable child IDs and sync methods. Do not mount/unmount mode labels during ordinary state refreshes.

- [ ] **Step 8: Add TCSS classes**

Create `tldw_chatbook/css/components/_workbench.tcss`:

```css
.workbench-frame {
    height: 100%;
    width: 100%;
    layout: vertical;
    overflow: hidden;
    background: $ds-surface-panel;
}

.workbench-header {
    height: auto;
    min-height: 2;
    padding: 0 1;
    border: solid $ds-grid-line;
}

.workbench-command-strip,
.workbench-mode-strip {
    height: 1;
    min-height: 1;
    padding: 0 1;
    layout: horizontal;
    background: $ds-surface-raised;
}

.workbench-action,
.workbench-mode {
    height: 1;
    min-height: 1;
    margin: 0 1 0 0;
}

.workbench-action.is-primary {
    text-style: bold;
}

.workbench-action.is-disabled {
    color: $ds-text-disabled;
}

.workbench-recovery-callout {
    height: auto;
    min-height: 0;
    padding: 0 1;
    border: solid $ds-status-warning;
    background: $ds-surface-panel;
}

.workbench-recovery-action {
    height: 1;
    min-height: 1;
    margin: 1 0 0 0;
}

.workbench-pane {
    height: 1fr;
    min-height: 0;
    border: solid $ds-grid-line;
}

.density-compact .workbench-header {
    min-height: 1;
    padding: 0 1;
}
```

Add `@import "./components/_workbench.tcss";` to `tldw_chatbook/css/main.tcss` after `_agentic_terminal.tcss`.

Add `"components/_workbench.tcss"` to `CSS_MODULES` after `"components/_agentic_terminal.tcss"`.

Regenerate:

Run: `python3 tldw_chatbook/css/build_css.py`

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` contains `MODULE: components/_workbench.tcss`.

- [ ] **Step 9: Export workbench APIs**

Update `tldw_chatbook/UI/Workbench/__init__.py`:

```python
"""Shared Textual workbench primitives for Chatbook destinations."""

from .workbench_state import (
    RecoveryState,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchPaneState,
    WorkbenchState,
)
from .workbench_widgets import (
    CommandStrip,
    DestinationHeader,
    ModeStrip,
    RecoveryCallout,
    StateBlock,
    WorkbenchActionRequested,
    WorkbenchFrame,
    WorkbenchPane,
)

__all__ = [
    "CommandStrip",
    "DestinationHeader",
    "ModeStrip",
    "RecoveryCallout",
    "RecoveryState",
    "StateBlock",
    "WorkbenchAction",
    "WorkbenchActionRequested",
    "WorkbenchFrame",
    "WorkbenchHeaderState",
    "WorkbenchMode",
    "WorkbenchPane",
    "WorkbenchPaneState",
    "WorkbenchState",
]
```

- [ ] **Step 10: Run widget/state/CSS tests**

Run: `pytest Tests/UI/test_workbench_state.py Tests/UI/test_workbench_widgets.py Tests/UI/test_master_shell_design_system_contract.py -q`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add tldw_chatbook/UI/Workbench/workbench_state.py \
  tldw_chatbook/UI/Workbench/workbench_widgets.py \
  tldw_chatbook/UI/Workbench/__init__.py \
  tldw_chatbook/css/components/_workbench.tcss \
  tldw_chatbook/css/main.tcss \
  tldw_chatbook/css/build_css.py \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_workbench_state.py \
  Tests/UI/test_workbench_widgets.py
git commit -m "Add shared Workbench UI primitives"
```

---

### Task 4: Focus, Footer, And Contextual Help

**Files:**
- Create: `tldw_chatbook/UI/Workbench/focus.py`
- Create: `tldw_chatbook/UI/Workbench/help.py`
- Modify: `tldw_chatbook/UI/Workbench/__init__.py`
- Modify: `tldw_chatbook/Widgets/AppFooterStatus.py:18-64`
- Modify: `tldw_chatbook/app.py:1126-1139`
- Modify: `tldw_chatbook/app.py:6727`
- Test: `Tests/UI/test_workbench_focus_help.py`
- Modify: `Tests/UI/test_app_footer_shortcut_context.py`

- [ ] **Step 1: Write failing focus model tests**

```python
from tldw_chatbook.UI.Workbench.focus import WorkbenchFocusRegistry


def test_focus_registry_cycles_visible_panes_only():
    registry = WorkbenchFocusRegistry(("context", "transcript", "inspector", "composer"))

    assert registry.next_after(None, hidden={"inspector"}) == "context"
    assert registry.next_after("context", hidden={"inspector"}) == "transcript"
    assert registry.next_after("transcript", hidden={"inspector"}) == "composer"
    assert registry.next_after("composer", hidden={"inspector"}) == "context"
```

- [ ] **Step 2: Write failing footer/help tests**

```python
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpState
from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchAction


def test_help_state_lists_visible_actions_not_palette_only():
    help_state = WorkbenchHelpState(
        route_id="chat",
        title="Console",
        actions=(WorkbenchAction("settings", "Settings"), WorkbenchAction("send", "Send")),
        shortcuts=(("F6", "next pane"), ("F1", "help")),
    )

    rendered = help_state.render_text()

    assert "Settings" in rendered
    assert "Send" in rendered
    assert "Ctrl+P" not in rendered
```

- [ ] **Step 3: Run focus/help tests to verify they fail**

Run: `pytest Tests/UI/test_workbench_focus_help.py -q`

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement focus registry**

Create `tldw_chatbook/UI/Workbench/focus.py`:

```python
"""Focus-order helpers for Workbench destinations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkbenchFocusRegistry:
    pane_order: tuple[str, ...]

    def next_after(self, current: str | None, *, hidden: set[str] | frozenset[str] = frozenset()) -> str | None:
        visible = tuple(pane for pane in self.pane_order if pane not in hidden)
        if not visible:
            return None
        if current not in visible:
            return visible[0]
        index = visible.index(current)
        return visible[(index + 1) % len(visible)]
```

- [ ] **Step 5: Implement help state and panel**

Create `tldw_chatbook/UI/Workbench/help.py`:

```python
"""Contextual help for visible Workbench actions."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Static
from textual.containers import Vertical

from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchAction


@dataclass(frozen=True)
class WorkbenchHelpState:
    route_id: str
    title: str
    actions: tuple[WorkbenchAction, ...] = ()
    shortcuts: tuple[tuple[str, str], ...] = ()

    def render_text(self) -> str:
        lines = [self.title]
        if self.actions:
            lines.append("Actions:")
            lines.extend(f"- {action.label}" for action in self.actions if not action.disabled)
        if self.shortcuts:
            lines.append("Shortcuts:")
            lines.extend(f"- {key}: {label}" for key, label in self.shortcuts)
        return "\n".join(lines)


class WorkbenchHelpPanel(ModalScreen[None]):
    def __init__(self, state: WorkbenchHelpState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="workbench-help-panel", classes="workbench-help-panel"):
            yield Static(self.state.render_text(), id="workbench-help-body")
            yield Button("Close", id="workbench-help-close", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "workbench-help-close":
            event.stop()
            self.dismiss(None)
```

- [ ] **Step 6: Extend footer context without breaking existing behavior**

Update `tldw_chatbook/Widgets/AppFooterStatus.py`:

- Keep `ShortcutContext` support as-is.
- Add a method:

```python
def set_workbench_shortcuts(self, *, source: str, shortcuts: tuple[tuple[str, str], ...]) -> None:
    context = ShortcutContext(
        source=source,
        actions=tuple(ShortcutAction(key, label) for key, label in shortcuts),
    )
    self.set_shortcut_context(context)
```

Update imports to include `ShortcutAction`.

Extend `Tests/UI/test_app_footer_shortcut_context.py`:

```python
def test_footer_renders_workbench_shortcuts():
    footer = AppFooterStatus()
    footer.compose()
```

Use an async app test like the existing tests; assert `F6 next pane` and `F1 help` render.

- [ ] **Step 7: Add app bindings and delegation**

Modify `tldw_chatbook/app.py` `BINDINGS`:

```python
Binding("f1", "show_workbench_help", "Help", show=True),
Binding("f6", "focus_next_workbench_pane", "Next Pane", show=True),
```

Add app actions near `action_quit`:

```python
def action_show_workbench_help(self) -> None:
    handler = getattr(self.screen, "action_show_workbench_help", None)
    if callable(handler):
        handler()
        return
    self.notify("No contextual help is available for this screen.", severity="information")


def action_focus_next_workbench_pane(self) -> None:
    handler = getattr(self.screen, "action_focus_next_workbench_pane", None)
    if callable(handler):
        handler()
        return
    self.notify("No workbench pane focus target is available.", severity="information")
```

- [ ] **Step 8: Run focus/help/footer tests**

Run: `pytest Tests/UI/test_workbench_focus_help.py Tests/UI/test_app_footer_shortcut_context.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/UI/Workbench/focus.py \
  tldw_chatbook/UI/Workbench/help.py \
  tldw_chatbook/UI/Workbench/__init__.py \
  tldw_chatbook/Widgets/AppFooterStatus.py \
  tldw_chatbook/app.py \
  Tests/UI/test_workbench_focus_help.py \
  Tests/UI/test_app_footer_shortcut_context.py
git commit -m "Add Workbench focus and contextual help"
```

---

### Task 5: Console Workbench State Adapter

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_workbench_state.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Test: `Tests/UI/test_console_workbench_contract.py`
- Existing context: `tldw_chatbook/Chat/console_display_state.py`
- Existing context: `tldw_chatbook/Chat/console_session_settings.py`

- [ ] **Step 1: Write failing Console adapter tests**

```python
from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_workbench_state import build_console_workbench_state


def test_console_workbench_state_exposes_core_actions_visibly():
    state = build_console_workbench_state(
        control_state=ConsoleControlState(
            provider_label="Provider: llama.cpp",
            model_label="Model: local-model",
            persona_label="Assistant: General",
            rag_label="RAG: off",
            sources_label="Sources: 0",
            tools_label="Tools: 0",
            approvals_label="Approvals: 0",
        ),
        provider_blocker_copy="",
        can_send=True,
        can_stop=False,
        can_save_chatbook=True,
    )

    labels = {action.label for action in state.actions}

    assert {"Settings", "Attach context", "Run Library RAG", "Save Chatbook", "Help"} <= labels
    assert state.header.title == "Console"
    assert state.recovery is None


def test_console_workbench_state_surfaces_provider_recovery():
    state = build_console_workbench_state(
        control_state=ConsoleControlState(
            provider_label="Provider: OpenAI",
            model_label="Model: --",
            persona_label="Assistant: General",
            rag_label="RAG: off",
            sources_label="Sources: 0",
            tools_label="Tools: 0",
            approvals_label="Approvals: 0",
        ),
        provider_blocker_copy="Provider setup needed: choose a model",
        provider_action_label="Choose model",
        can_send=False,
        can_stop=False,
        can_save_chatbook=False,
    )

    assert state.recovery is not None
    assert "choose a model" in state.recovery.body.lower()
    assert state.recovery.action is not None
    assert state.recovery.action.label == "Choose model"
```

- [ ] **Step 2: Run adapter tests to verify they fail**

Run: `pytest Tests/UI/test_console_workbench_contract.py::test_console_workbench_state_exposes_core_actions_visibly Tests/UI/test_console_workbench_contract.py::test_console_workbench_state_surfaces_provider_recovery -q`

Expected: FAIL with missing `console_workbench_state`.

- [ ] **Step 3: Implement adapter**

Create `tldw_chatbook/Widgets/Console/console_workbench_state.py`:

```python
"""Console adapters for shared Workbench UI state."""

from __future__ import annotations

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.UI.Workbench.workbench_state import (
    Density,
    RecoveryState,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchPaneState,
    WorkbenchState,
)


def build_console_workbench_state(
    *,
    control_state: ConsoleControlState,
    provider_blocker_copy: str = "",
    provider_action_label: str = "Open Settings",
    can_send: bool = False,
    can_stop: bool = False,
    can_save_chatbook: bool = False,
    density: str = "normal",
) -> WorkbenchState:
    """Return a shared Workbench state snapshot for Console."""

    blocker = provider_blocker_copy.strip()
    workbench_density: Density = "compact" if density == "compact" else "normal"
    actions = (
        WorkbenchAction("new-tab", "New tab", "Create a Console tab"),
        WorkbenchAction("settings", "Settings", "Configure provider, model, tools, and generation"),
        WorkbenchAction("attach-context", "Attach context", "Stage Library or workspace context"),
        WorkbenchAction("run-library-rag", "Run Library RAG", "Search Library evidence before sending"),
        WorkbenchAction("save-chatbook", "Save Chatbook", "Save this run as a Chatbook", disabled=not can_save_chatbook),
        WorkbenchAction("send", "Send", "Send composer draft", disabled=not can_send, primary=can_send),
        WorkbenchAction("stop", "Stop", "Stop active generation", disabled=not can_stop),
        WorkbenchAction("help", "Help", "Show visible Console actions and shortcuts"),
    )
    modes = (
        WorkbenchMode("provider", control_state.provider_label, active=True, status="blocked" if blocker else "ready"),
        WorkbenchMode("model", control_state.model_label, status="blocked" if blocker else "ready"),
        WorkbenchMode("persona", control_state.persona_label),
        WorkbenchMode("rag", control_state.rag_label),
        WorkbenchMode("sources", control_state.sources_label),
        WorkbenchMode("tools", control_state.tools_label),
        WorkbenchMode("approvals", control_state.approvals_label),
    )
    recovery = None
    if blocker:
        recovery = RecoveryState(
            title="Console setup blocked",
            body=f"{blocker}\nImpact: Send is blocked until setup is finished.",
            action=WorkbenchAction("provider-recovery", provider_action_label, primary=True),
        )
    return WorkbenchState(
        route_id="chat",
        density=workbench_density,
        header=WorkbenchHeaderState(
            title="Console",
            subtitle="Chat, source handoffs, live runs, and control actions.",
            status="blocked" if blocker else "ready",
            density=workbench_density,
        ),
        modes=modes,
        actions=actions,
        panes=(
            WorkbenchPaneState("context", "Context"),
            WorkbenchPaneState("transcript", "Transcript"),
            WorkbenchPaneState("inspector", "Inspector"),
            WorkbenchPaneState("composer", "Composer"),
        ),
        recovery=recovery,
    )
```

- [ ] **Step 4: Export adapter**

Update `tldw_chatbook/Widgets/Console/__init__.py`:

```python
from .console_workbench_state import build_console_workbench_state
```

Add it to `__all__`.

- [ ] **Step 5: Run adapter tests**

Run: `pytest Tests/UI/test_console_workbench_contract.py -q`

Expected: adapter tests PASS. Other tests in the new file may still fail until later tasks if they are marked for integration; keep this file focused at this point.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workbench_state.py \
  tldw_chatbook/Widgets/Console/__init__.py \
  Tests/UI/test_console_workbench_contract.py
git commit -m "Add Console Workbench state adapter"
```

---

### Task 6: Refactor Console Shell To Shared Workbench Primitives

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:118-133`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:2582-2916`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:3085-3368`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Test: `Tests/UI/test_console_workbench_contract.py`
- Existing tests to keep green: `Tests/UI/test_console_persistent_rails.py`, `Tests/UI/test_console_internals_decomposition.py`, `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_console_workspace_context_rail.py`, `Tests/UI/test_console_live_work_handoffs.py`

- [ ] **Step 1: Add failing visible-control integration tests**

Append to `Tests/UI/test_console_workbench_contract.py`:

```python
import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _is_displayed(widget) -> bool:
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = getattr(current, "parent", None)
    return True


@pytest.mark.asyncio
async def test_console_core_controls_are_visible_without_command_palette():
    app = _build_test_app(initial_tab="chat")

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-shell")

        for selector in (
            "#workbench-action-settings",
            "#workbench-action-attach-context",
            "#workbench-action-run-library-rag",
            "#workbench-action-help",
            "#console-native-composer",
        ):
            widget = app.screen.query_one(selector)
            assert _is_displayed(widget), selector


@pytest.mark.asyncio
async def test_console_recovery_action_is_visible_when_provider_setup_blocks_send():
    app = _build_test_app(initial_tab="chat")
    app.app_config = {"chat_defaults": {"provider": "OpenAI", "model": ""}, "api_settings": {"openai": {"api_key": ""}}}
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-shell")

        recovery = app.screen.query_one("#workbench-recovery-callout")
        assert _is_displayed(recovery)
        recovery_text = " ".join(
            getattr(child.renderable, "plain", str(getattr(child, "renderable", "")))
            for child in recovery.query("Static")
        )
        assert "Send is blocked" in recovery_text


@pytest.mark.asyncio
async def test_console_recovery_action_button_is_visible_and_actionable():
    app = _build_test_app(initial_tab="chat")
    app.app_config = {"chat_defaults": {"provider": "OpenAI", "model": ""}, "api_settings": {"openai": {"api_key": ""}}}
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-shell")
        await _wait_for_selector(app.screen, pilot, "#workbench-recovery-action")

        action = app.screen.query_one("#workbench-recovery-action")
        assert _is_displayed(action)
        assert not action.disabled
        await pilot.click("#workbench-recovery-action")
        await pilot.pause()

        assert app.screen.query("#console-settings-modal") or app.screen.query("#settings-screen")
```

If the recovery path opens a different existing modal or navigates to Settings, update the last assertion to the actual visible destination, but keep the assertion that clicking the visible recovery action changes UI state.

- [ ] **Step 2: Run the new Console workbench tests to verify they fail**

Run: `pytest Tests/UI/test_console_workbench_contract.py -q`

Expected: FAIL because `#workbench-action-*` selectors are not mounted in Console yet.

- [ ] **Step 3: Import workbench APIs into ChatScreen**

Modify `tldw_chatbook/UI/Screens/chat_screen.py` imports:

```python
from ...UI.Workbench import (
    CommandStrip,
    DestinationHeader,
    ModeStrip,
    RecoveryCallout,
    WorkbenchActionRequested,
    WorkbenchFrame,
    WorkbenchPane,
)
from ...Widgets.Console.console_workbench_state import build_console_workbench_state
```

Remove unused local imports only after tests confirm they are no longer needed.

- [ ] **Step 4: Add a ChatScreen helper that builds Console workbench state**

Near `_console_mode_summary`, add:

```python
def _build_console_workbench_state(self, control_state: ConsoleControlState):
    blocker_copy = self._console_provider_blocker_copy()
    action_label, _action_target, _action_tooltip = self._console_provider_recovery_action()
    composer = self._console_composer_or_none()
    has_draft = bool(composer and composer.draft_text().strip())
    controller = self._console_chat_controller
    run_active = bool(controller and controller.run_state.status == ConsoleRunStatus.RUNNING)
    can_send = has_draft and not bool(self._console_setup_blocked_reason())
    return build_console_workbench_state(
        control_state=control_state,
        provider_blocker_copy=blocker_copy,
        provider_action_label=action_label,
        can_send=can_send,
        can_stop=run_active,
        can_save_chatbook=self._console_chatbook_action_available(),
    )
```

Adjust the exact `ConsoleRunStatus` comparison to match existing enum values in `console_chat_models.py`.

- [ ] **Step 5: Replace hidden title/status/control region in `compose_content`**

In `compose_content`, before `with Vertical(id="console-shell")`, build:

```python
workbench_state = self._build_console_workbench_state(control_state)
```

Then replace the title/status/control block with:

```python
with Vertical(id="console-shell", classes="workbench-frame console-workbench-frame"):
    yield DestinationHeader(
        workbench_state.header,
        id="console-workbench-header",
        classes="workbench-header",
    )
    yield ModeStrip(
        workbench_state.modes,
        id="console-workbench-mode-strip",
        classes="workbench-mode-strip",
    )
    yield CommandStrip(
        workbench_state.actions,
        id="console-workbench-command-strip",
        classes="workbench-command-strip",
    )
    yield RecoveryCallout(
        workbench_state.recovery,
        id="workbench-recovery-callout",
        classes="workbench-recovery-callout",
    )
```

Keep the existing `#console-workspace-grid`, `#console-left-rail`, `#console-main-column`, `#console-right-rail`, and `#console-native-composer` structure below this header. This preserves the layout framing while changing the widgets in use.

- [ ] **Step 6: Keep backward-compatible selectors during parity**

Do not remove these selectors yet:

- `#console-title`
- `#console-mode-bar`
- `#console-control-bar`
- `#console-provider-recovery-strip`
- `#console-provider-blocker`
- `#console-open-provider-settings`

If a legacy selector has no visible role, keep a zero-height compatibility widget only until existing tests are updated. Document each retained compatibility selector in a comment in `compose_content`.

- [ ] **Step 7: Route workbench action messages**

Add handler in `ChatScreen`:

```python
@on(WorkbenchActionRequested)
def on_console_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
    event.stop()
    action_id = event.action_id
    if action_id == "new-tab":
        self.run_worker(self._create_native_console_session_from_active_context(), exclusive=True)
    elif action_id == "settings":
        self.run_worker(self._open_console_settings(focus_model=False), exclusive=True)
    elif action_id == "attach-context":
        self._set_console_rail_preference(left_open=True)
    elif action_id == "run-library-rag":
        self._run_console_library_rag_from_visible_action()
    elif action_id == "save-chatbook":
        self._save_console_chatbook_from_visible_action()
    elif action_id == "send":
        self.action_console_send_message()
    elif action_id == "stop":
        self.action_stop_generation()
    elif action_id == "help":
        self.action_show_workbench_help()
    elif action_id == "provider-recovery":
        self.run_worker(self._open_console_provider_recovery(), exclusive=True)
```

The Workbench handler must call Console helpers directly. Do not route new Workbench actions through hidden compatibility widgets or `.press()`; the important contract is one visible action path per core workflow.

- [ ] **Step 8: Add or update direct helper methods for actions**

Implement direct helpers where missing:

- `_open_console_provider_recovery()`
- `_open_console_settings(*, focus_model: bool = False)`
- `_run_console_library_rag_from_visible_action()`
- `_save_console_chatbook_from_visible_action()`

Refactor the current button event handlers to become adapters over the same helpers:

- `on_console_settings_open` stops the event, derives the model-focus flag from the existing button/recovery logic, then awaits `_open_console_settings(focus_model=...)`.
- `handle_console_run_library_rag` stops the event, then calls `_run_console_library_rag_from_visible_action()`.
- `handle_console_save_chatbook` stops the event, then calls `_save_console_chatbook_from_visible_action()`.
- `handle_console_open_provider_settings` stops the event, then awaits `_open_console_provider_recovery()`.

These helpers should use existing methods and state, not duplicate provider/RAG/save logic.

- [ ] **Step 9: Update ConsoleControlBar**

In `tldw_chatbook/Widgets/Console/console_control_bar.py`, remove `_hide_layout_widget()` from normal rendering. Either:

- convert it to a visible compact summary used by `ModeStrip`, or
- leave it as a compatibility-only widget with a docstring comment that names the parity test guarding its eventual removal.

The preferred outcome is no hidden provider/model/persona/RAG/source/tool/approval labels in the reference Console path.

- [ ] **Step 10: Run Console workbench contract tests**

Run: `pytest Tests/UI/test_console_workbench_contract.py -q`

Expected: PASS.

- [ ] **Step 11: Run existing Console regression tests**

Run: `pytest Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_live_work_handoffs.py -q`

Expected: PASS. If existing tests assert hidden controls, update them to assert visible workbench controls or documented compatibility selectors.

- [ ] **Step 12: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/Widgets/Console/console_control_bar.py \
  tldw_chatbook/Widgets/Console/console_session_surface.py \
  tldw_chatbook/Widgets/Console/console_composer_bar.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_persistent_rails.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_live_work_handoffs.py
git commit -m "Refactor Console onto Workbench primitives"
```

---

### Task 7: Console Focus, Help, Density, And Discoverability Gates

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:3394-3425`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:4866-4989`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/css/components/_workbench.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_workbench_contract.py`
- Test: `Tests/UI/test_workbench_focus_help.py`

- [ ] **Step 1: Add failing F6 focus test**

Append to `Tests/UI/test_console_workbench_contract.py`:

```python
@pytest.mark.asyncio
async def test_console_f6_cycles_visible_workbench_panes():
    app = _build_test_app(initial_tab="chat")

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-shell")

        await pilot.press("f6")
        assert app.screen.focused is not None
        first_id = app.screen.focused.id
        await pilot.press("f6")
        second_id = app.screen.focused.id

        assert first_id != second_id
        assert {first_id, second_id} <= {
            "console-left-rail",
            "console-transcript-surface",
            "console-right-rail",
            "console-native-composer",
        }
```

Also append a route-switch focus regression. Import `NavigateToScreen` from `tldw_chatbook.UI.Navigation.main_navigation` if the test file does not already import it:

```python
@pytest.mark.asyncio
async def test_console_route_switch_restores_workbench_focus():
    app = _build_test_app(initial_tab="chat")

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-shell")
        app.screen.query_one("#console-native-composer").focus()

        await app.handle_screen_navigation(NavigateToScreen("settings"))
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await _wait_for_selector(app.screen, pilot, "#console-shell")

        focused_id = getattr(getattr(app, "focused", None), "id", None)
        if focused_id is None:
            focused_id = getattr(getattr(app.screen, "focused", None), "id", None)

        assert focused_id in {
            "console-left-rail",
            "console-transcript-surface",
            "console-right-rail",
            "console-native-composer",
        }
```

- [ ] **Step 2: Add failing help test**

```python
@pytest.mark.asyncio
async def test_console_f1_help_lists_visible_actions():
    app = _build_test_app(initial_tab="chat")

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-shell")
        await pilot.press("f1")
        await _wait_for_selector(app, pilot, "#workbench-help-panel")

        body = app.query_one("#workbench-help-body")
        text = getattr(body.renderable, "plain", str(body.renderable))
        assert "Settings" in text
        assert "Attach context" in text
        assert "Run Library RAG" in text
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_workbench_contract.py::test_console_f6_cycles_visible_workbench_panes Tests/UI/test_console_workbench_contract.py::test_console_route_switch_restores_workbench_focus Tests/UI/test_console_workbench_contract.py::test_console_f1_help_lists_visible_actions -q`

Expected: FAIL because Console does not implement workbench focus/help actions yet.

- [ ] **Step 4: Register Console footer shortcuts on mount**

In `ChatScreen.on_mount`, after parent setup:

```python
footer = self.app.query_one(AppFooterStatus)
footer.set_workbench_shortcuts(
    source="console",
    shortcuts=(
        ("F6", "next pane"),
        ("F1", "help"),
        ("Enter", "send"),
        ("Ctrl+P", "palette"),
    ),
)
```

Use `try/except QueryError` so tests without footer do not fail.

In `on_unmount`, clear with `footer.clear_shortcut_context("console")`.

- [ ] **Step 5: Implement Console focus action**

Add a `WorkbenchFocusRegistry` instance or helper:

```python
CONSOLE_FOCUS_REGISTRY = WorkbenchFocusRegistry(
    ("console-left-rail", "console-transcript-surface", "console-right-rail", "console-native-composer")
)
```

Implement:

```python
def action_focus_next_workbench_pane(self) -> None:
    hidden = {
        widget_id
        for widget_id in ("console-left-rail", "console-right-rail")
        if not self._is_console_widget_displayed(widget_id)
    }
    current = getattr(self.focused, "id", None)
    next_id = CONSOLE_FOCUS_REGISTRY.next_after(current, hidden=hidden)
    if not next_id:
        return
    try:
        self.query_one(f"#{next_id}").focus()
    except QueryError:
        return
```

Add `_is_console_widget_displayed(widget_id: str) -> bool` based on existing test helper logic.

When Console becomes active after a route switch, restore focus to the last visible Workbench pane when possible; otherwise focus `#console-native-composer` if it is mounted, then `#console-transcript-surface`.

- [ ] **Step 6: Implement Console help action**

Use `WorkbenchHelpState` and `WorkbenchHelpPanel`:

```python
def action_show_workbench_help(self) -> None:
    control_state = self._build_console_control_state(self._pending_console_launch_context)
    state = self._build_console_workbench_state(control_state)
    self.app.push_screen(
        WorkbenchHelpPanel(
            WorkbenchHelpState(
                route_id="chat",
                title="Console",
                actions=state.actions,
                shortcuts=(("F6", "next pane"), ("F1", "help"), ("Enter", "send"), ("Ctrl+P", "palette")),
            )
        )
    )
```

- [ ] **Step 7: Add density classes**

Add a simple source for density:

- app config key: `appearance.ui_density`, default `normal`.
- helper in ChatScreen: `_workbench_density()` returns `"compact"` or `"normal"`.
- pass density into `build_console_workbench_state`.
- add class to `#console-shell`: `density-compact` or `density-normal`.

Do not add a settings UI in this task unless an existing appearance setting already exists.

- [ ] **Step 8: Add density test**

```python
def test_workbench_css_contains_normal_and_compact_density():
    css = Path("tldw_chatbook/css/components/_workbench.tcss").read_text(encoding="utf-8")

    assert ".density-compact" in css
    assert ".density-normal" in css
```

- [ ] **Step 9: Rebuild CSS**

Run: `python3 tldw_chatbook/css/build_css.py`

Expected: generated CSS includes `density-compact` and `density-normal` workbench rules.

- [ ] **Step 10: Run focus/help/density tests**

Run: `pytest Tests/UI/test_console_workbench_contract.py Tests/UI/test_workbench_focus_help.py -q`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/Widgets/Console/console_composer_bar.py \
  tldw_chatbook/css/components/_workbench.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_workbench_focus_help.py
git commit -m "Add Console Workbench focus help and density gates"
```

---

### Task 8: Verification, Documentation, And Task Closeout

**Files:**
- Create: `Docs/Design/chatbook-workbench-ui-system.md`
- Create: `Tests/UI/test_workbench_visual_snapshots.py`
- Create: `Tests/UI/test_console_workbench_parity_matrix.py`
- Modify: `Docs/Design/master-shell-design-system-contract.md`
- Modify: `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`
- Modify: `backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md`
- Verify: all files touched in Tasks 1-7

- [ ] **Step 1: Write the Workbench design doc**

Create `Docs/Design/chatbook-workbench-ui-system.md`:

```markdown
# Chatbook Workbench UI System

The Workbench UI System is the shared Textual-native frame for Chatbook destinations.

## Principles

- Stable composition: compose regions once, sync state into mounted widgets.
- Visible workflows: core actions must be reachable without the command palette.
- Explicit state: headers, modes, panes, recovery, and actions are snapshots.
- Domain ownership: shared widgets do not query databases or provider services.
- Responsiveness gates: route migrations must track heartbeat lag, workers, timers, and mount churn.

## Required Visible Actions

Console must visibly expose provider/model settings, send/stop, attach context, Library RAG, inspector review, save Chatbook, help, and recovery actions.

## Migration Rule

Every route in `Constants.py`, `shell_destinations.py`, and `screen_registry.py` must have a migration owner in `tldw_chatbook/UI/Workbench/route_inventory.py`.
```

- [ ] **Step 2: Link design doc from master shell contract**

Add a short section to `Docs/Design/master-shell-design-system-contract.md`:

```markdown
## Workbench UI System

See `Docs/Design/chatbook-workbench-ui-system.md` for shared destination workbench widgets, visible-action rules, route migration ownership, and responsiveness gates.
```

- [ ] **Step 3: Add Workbench visual snapshot tests**

Create `Tests/UI/test_workbench_visual_snapshots.py` using the existing `export_screenshot` pattern from `Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py`:

```python
from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


BROKEN_TEXT_PATTERNS = ("Traceback", "Unhandled exception", "Unable to mount", "Internal Error")
RAW_OBJECT_REPR = re.compile(r"<[\w.]+ object at 0x[0-9a-fA-F]+>")


def _setting_with_density(density: str):
    def _get_cli_setting(section: str, key: str, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        if section == "appearance" and key == "ui_density":
            return density
        return default
    return _get_cli_setting


def _assert_svg_healthy(svg: str) -> None:
    assert "<svg" in svg
    assert "</svg>" in svg
    assert len(svg) > 1_000
    for broken in BROKEN_TEXT_PATTERNS:
        assert broken not in svg
    assert RAW_OBJECT_REPR.search(svg) is None


@pytest.mark.parametrize("density", ("normal", "compact"))
@pytest.mark.asyncio
async def test_console_workbench_normal_and_compact_snapshots(density: str):
    app = _build_test_app(initial_tab="chat")
    app.app_config.setdefault("appearance", {})["ui_density"] = density

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_setting_with_density(density)):
        async with app.run_test(size=(140, 42)) as pilot:
            await _wait_for_selector(app.screen, pilot, "#console-shell")

            shell = app.screen.query_one("#console-shell")
            assert shell.has_class(f"density-{density}")
            _assert_svg_healthy(app.export_screenshot(title=f"Console Workbench {density}", simplify=True))


@pytest.mark.asyncio
async def test_console_workbench_command_palette_snapshot():
    app = _build_test_app(initial_tab="chat")

    async with app.run_test(size=(140, 42)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-shell")
        await pilot.press("ctrl+p")
        await pilot.pause()

        _assert_svg_healthy(app.export_screenshot(title="Console Workbench Command Palette", simplify=True))


@pytest.mark.asyncio
async def test_console_workbench_focus_state_snapshot():
    app = _build_test_app(initial_tab="chat")

    async with app.run_test(size=(140, 42)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-shell")
        settings_action = app.screen.query_one("#workbench-action-settings", Button)
        settings_action.focus()
        await pilot.pause()

        assert app.focused is settings_action
        _assert_svg_healthy(app.export_screenshot(title="Console Workbench Focus State", simplify=True))
```

Run: `pytest Tests/UI/test_workbench_visual_snapshots.py -q`

Expected: PASS after Task 7.

- [ ] **Step 4: Add Console parity matrix tests**

Create `Tests/UI/test_console_workbench_parity_matrix.py`:

```python
from __future__ import annotations

from pathlib import Path


CONSOLE_PARITY_MATRIX: dict[str, tuple[str, ...]] = {
    "streaming_send_stop": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_stop_interrupts_stream_and_keeps_partial_message_visible",
        "Tests/UI/test_console_native_chat_flow.py::test_console_duplicate_send_during_stream_does_not_break_stop_control",
    ),
    "non_streaming_fallback": (
        "Tests/Chat/test_console_provider_gateway.py::test_llamacpp_stream_chat_falls_back_to_non_streaming_when_stream_rejected",
        "Tests/Chat/test_console_provider_gateway.py::test_stream_chat_non_streaming_resolution_yields_completion_once",
    ),
    "provider_model_selection": (
        "Tests/UI/test_console_session_settings.py",
        "Tests/Chat/test_console_provider_support.py",
        "Tests/Chat/test_console_provider_endpoints.py",
    ),
    "multi_session_tabs": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_promotes_active_native_session_in_workspace_rail",
        "Tests/UI/test_console_native_chat_flow.py::test_console_close_tab_with_messages_shows_confirmation",
        "Tests/UI/test_console_native_chat_flow.py::test_console_native_active_tab_title_opens_rename_modal",
        "Tests/Chat/test_console_chat_store.py::test_console_sessions_store_independent_settings_snapshots",
    ),
    "retry_regenerate_continue": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_failed_stream_renders_inline_retry_and_recovers",
        "Tests/UI/test_console_native_chat_flow.py::test_console_continue_action_streams_new_message_from_selected_turn",
        "Tests/UI/test_console_native_chat_flow.py::test_console_regenerate_action_streams_selected_variant",
    ),
    "edit_delete_copy_feedback": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_copy_action_uses_app_clipboard",
        "Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_edit_action_opens_modal_and_saves_content",
        "Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_delete_action_removes_message_from_transcript",
        "Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_feedback_action_records_rating",
    ),
    "attachments_images": (
        "Tests/UI/test_chat_image_attachment.py",
        "Tests/Chat/test_chat_functions.py::TestChatFunction::test_chat_with_image_and_rag",
    ),
    "tool_call_visibility": (
        "Tests/UI/test_console_internals_decomposition.py::test_console_run_inspector_exposes_pending_approval_and_chatbook_artifact_actions",
        "Tests/integration/test_chat_tool_flow.py",
    ),
    "workspace_and_live_work_handoffs": (
        "Tests/UI/test_console_workspace_context_rail.py",
        "Tests/UI/test_console_live_work_handoffs.py",
    ),
    "staged_context_sources": (
        "Tests/UI/test_console_workspace_context_rail.py",
        "Tests/UI/test_console_live_work_handoffs.py",
        "Tests/Chat/test_console_chat_controller.py::test_blocked_workspace_source_preserves_draft_and_skips_provider_call",
    ),
    "recovery_states": (
        "Tests/UI/test_console_native_chat_flow.py::test_console_setup_required_state_groups_recovery_and_action_copy",
        "Tests/UI/test_console_native_chat_flow.py::test_console_setup_blocked_send_adds_durable_transcript_recovery_feedback",
        "Tests/UI/test_console_native_chat_flow.py::test_console_failed_stream_renders_inline_retry_and_recovers",
    ),
    "persistence_behavior": (
        "Tests/Chat/test_console_chat_store.py",
        "Tests/Chat/test_console_chat_controller.py::test_retry_failed_message_streams_replacement_from_original_turn",
    ),
    "visible_workbench_actions": (
        "Tests/UI/test_console_workbench_contract.py::test_console_core_controls_are_visible_without_command_palette",
        "Tests/UI/test_console_workbench_contract.py::test_console_recovery_action_button_is_visible_and_actionable",
    ),
    "command_palette_duplicates": (
        "Tests/UI/test_command_palette_basic.py",
        "Tests/UI/test_command_palette_shell_routes.py",
        "Tests/UI/test_command_palette_providers.py",
    ),
}


def test_console_parity_matrix_covers_required_behaviors():
    assert set(CONSOLE_PARITY_MATRIX) == {
        "streaming_send_stop",
        "non_streaming_fallback",
        "provider_model_selection",
        "multi_session_tabs",
        "retry_regenerate_continue",
        "edit_delete_copy_feedback",
        "attachments_images",
        "tool_call_visibility",
        "workspace_and_live_work_handoffs",
        "staged_context_sources",
        "recovery_states",
        "persistence_behavior",
        "visible_workbench_actions",
        "command_palette_duplicates",
    }


def test_console_parity_matrix_references_existing_test_files():
    for tests in CONSOLE_PARITY_MATRIX.values():
        for test_ref in tests:
            file_part = test_ref.split("::", 1)[0]
            assert Path(file_part).exists(), test_ref
```

Run: `pytest Tests/UI/test_console_workbench_parity_matrix.py -q`

Expected: PASS.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
pytest Tests/UI/test_workbench_route_inventory.py \
  Tests/UI/test_ui_responsiveness.py \
  Tests/UI/test_ui_responsiveness_artifacts.py \
  Tests/UI/test_workbench_state.py \
  Tests/UI/test_workbench_widgets.py \
  Tests/UI/test_workbench_focus_help.py \
  Tests/UI/test_workbench_visual_snapshots.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_workbench_parity_matrix.py \
  Tests/UI/test_app_footer_shortcut_context.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Run Console regression and parity set**

Run:

```bash
pytest Tests/UI/test_console_persistent_rails.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_chat_image_attachment.py \
  Tests/UI/test_command_palette_basic.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_display_state.py \
  Tests/Chat/test_console_provider_support.py \
  Tests/Chat/test_console_provider_endpoints.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_chat_functions.py::TestChatFunction::test_chat_with_image_and_rag \
  Tests/integration/test_chat_tool_flow.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Run route/navigation smoke tests**

Run:

```bash
pytest Tests/UI/test_shell_destinations.py \
  Tests/UI/test_screen_navigation.py \
  Tests/UI/test_master_shell_navigation.py \
  Tests/UI/test_command_palette_shell_routes.py \
  -q
```

Expected: PASS.

- [ ] **Step 8: Record QA evidence**

Append to `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`:

```markdown
## Verification

- Workbench route inventory:
- Responsiveness monitor:
- Workbench widgets:
- Workbench visual snapshots:
- Console workbench contract:
- Console parity matrix:
- Console regression set:
- Navigation smoke:
- Route-switch soak:
- Responsiveness artifacts:
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/ui_heartbeat.log`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/worker_snapshot.log`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/timer_registry.log`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/mount_churn_summary.log`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/route_switch_soak_result.txt`

## Residual Risks

- The automated soak uses synthetic Console interactions; a real provider streaming soak can be added as follow-up evidence when a provider endpoint is available.
- Later destinations are tracked by route owner but not migrated in this task.
```

Fill each bullet with the command and pass/fail output summary.

- [ ] **Step 9: Update Backlog acceptance criteria**

In `backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md`, mark each AC complete only after the associated verification passes:

```markdown
- [x] ADR records the Chatbook Workbench UI System decision
- [x] Responsiveness instrumentation captures event-loop, worker, timer, and mount churn baseline
- [x] Shared workbench primitives support stable composition, visible state, and normal plus compact density
- [x] Console replacement reaches feature parity before legacy retirement
- [x] Core Console workflow is completable without command palette
- [x] Route inventory coverage is tracked for future migration owners
- [x] Snapshot, interaction, and soak verification gates are defined and runnable
```

- [ ] **Step 10: Add Backlog implementation notes**

Add `## Implementation Notes`:

```markdown
Implemented the first Workbench UI redesign slice: route coverage, ADR-linked migration ownership, responsiveness instrumentation, shared workbench primitives, focus/help conventions, and Console as the reference implementation. Console now exposes core workflow controls visibly in the workbench frame while preserving the existing left context rail, transcript surface, inspector rail, and composer layout framing. Later destination migrations are intentionally left to separate scoped tasks using the route-owner coverage introduced here.

Modified files include `tldw_chatbook/UI/Workbench/*`, `tldw_chatbook/Utils/ui_responsiveness.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, Console widgets, workbench TCSS, Console/UI tests, and Workbench design docs.
```

- [ ] **Step 11: Mark task Done using Backlog CLI only after all DoD items pass**

Run: `backlog task edit 139 -s Done`

Expected: task status becomes `Done`.

- [ ] **Step 12: Final commit**

```bash
git add Docs/Design/chatbook-workbench-ui-system.md \
  Docs/Design/master-shell-design-system-contract.md \
  Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md \
  "backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md"
git commit -m "Document Workbench UI foundation verification"
```

---

## Final Verification Commands

Run these before claiming completion:

```bash
pytest Tests/UI/test_workbench_route_inventory.py \
  Tests/UI/test_ui_responsiveness.py \
  Tests/UI/test_ui_responsiveness_artifacts.py \
  Tests/UI/test_workbench_state.py \
  Tests/UI/test_workbench_widgets.py \
  Tests/UI/test_workbench_focus_help.py \
  Tests/UI/test_workbench_visual_snapshots.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_workbench_parity_matrix.py \
  Tests/UI/test_app_footer_shortcut_context.py \
  -q
```

```bash
pytest Tests/UI/test_console_persistent_rails.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_chat_image_attachment.py \
  Tests/UI/test_command_palette_basic.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_display_state.py \
  Tests/Chat/test_console_provider_support.py \
  Tests/Chat/test_console_provider_endpoints.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_chat_functions.py::TestChatFunction::test_chat_with_image_and_rag \
  Tests/integration/test_chat_tool_flow.py \
  -q
```

```bash
pytest Tests/UI/test_shell_destinations.py \
  Tests/UI/test_screen_navigation.py \
  Tests/UI/test_master_shell_navigation.py \
  Tests/UI/test_command_palette_shell_routes.py \
  -q
```

```bash
python3 tldw_chatbook/css/build_css.py
git diff --check
```

```bash
python3 Tests/UI/run_workbench_soak.py --output Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts --route-switches 6 --idle-seconds 10
test -s Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/ui_heartbeat.log
test -s Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/worker_snapshot.log
test -s Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/timer_registry.log
test -s Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/mount_churn_summary.log
test -s Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/route_switch_soak_result.txt
```

If the full project suite is practical in the implementation environment, also run `pytest -q`. If it is not practical, document why in `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md` and include the targeted suites above plus `git diff --check`.

## Rollback Plan

If Console regressions appear after Task 6:

1. Keep `tldw_chatbook/UI/Workbench/*`, route inventory, and responsiveness instrumentation.
2. Revert only the Console `compose_content` integration and `console_control_bar.py` changes from the latest task commit.
3. Keep tests documenting the desired visible-control behavior, marked as expected failures only if the user approves a pause.
4. Do not remove ADR-011 or TASK-141; update implementation notes with the rollback reason.

## Execution Notes

- Use `superpowers:subagent-driven-development` for implementation unless the user chooses inline execution.
- Use `superpowers:verification-before-completion` before any final completion claim.
- Keep commits task-sized. Do not combine instrumentation, primitives, and Console refactor into one commit.
- Do not migrate non-Console destinations in this task.
