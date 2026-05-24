# Console Persistent Rails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Console persistent-rails UX: first-start Console shows the left Context rail open and the right Inspector rail collapsed, both rails are user-toggleable through visible in-layout handles, rail state persists per workspace/session, collapsed handles show deterministic badges, and existing Console chat/provider/transcript/composer behavior keeps passing.

**Architecture:** Add a pure `console_rail_state` contract beside existing Console display-state contracts, add a small reusable `ConsoleRailHandle` widget, then wire `ChatScreen` to mount both rails and handles while toggling `display`/width state instead of remounting the composer or native chat surface. Persist only rail booleans in the existing `[console.rail_state]` config section using deterministic workspace/session keys, and copy temporary session preferences to the durable conversation key when a conversation id becomes available.

**Tech Stack:** Python 3.11+, Textual, TCSS, existing `ChatScreen`/Console widgets, existing `save_setting_to_cli_config`, pytest, textual mounted tests, textual-web/CDP screenshot QA.

---

## Source Material

- Approved design spec: `Docs/superpowers/specs/2026-05-24-console-persistent-rails-design.md`
- Current Console composition: `tldw_chatbook/UI/Screens/chat_screen.py`
- Console display-state contracts: `tldw_chatbook/Chat/console_display_state.py`
- Native Console chat contracts: `tldw_chatbook/Chat/console_chat_models.py`
- Console widgets: `tldw_chatbook/Widgets/Console/`
- Workspace context display state: `tldw_chatbook/Workspaces/display_state.py`
- Console styles: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Textual-web/CDP runbook: `Docs/superpowers/qa/product-maturity/screen-qa/textual-web-cdp-debugging.md`
- Current Console tests:
  - `Tests/Chat/test_console_display_state.py`
  - `Tests/UI/test_console_internals_decomposition.py`
  - `Tests/UI/test_console_native_transcript.py`
  - `Tests/UI/test_console_workspace_context_rail.py`
  - `Tests/UI/test_console_native_chat_flow.py`
- Screenshot approval rule: `Docs/superpowers/handoffs/2026-05-08-ui-screenshot-approval-workflow-handoff.md`

## Current Constraints

- Do not refactor the native chat core, provider gateway, transcript message actions, tab model, staged-context tray, workspace tray, run inspector behavior, or composer behavior except where required to expose rail collapse state.
- Do not make the right Inspector auto-open for provider blocks, failed runs, approvals, tools, source readiness, or staged context.
- Do not hide the central provider recovery strip behind the collapsed Inspector. `Open Settings` must remain reachable in the center lane.
- Do not use a full-screen recompose for rail toggles if it drops composer drafts, transcript selection, streaming state, or native session state.
- Persist only rail booleans, not rendered widths, badges, labels, or compact responsive overrides.
- Compact-width forced right collapse must not overwrite a stored `right_open=True` preference.
- Existing untracked files in the working tree are unrelated and must remain untouched.

## File Structure

### New Files

- `tldw_chatbook/Chat/console_rail_state.py`
  - Pure dataclasses and helper functions for rail preferences, deterministic persistence keys, compact-width effective state, and badge derivation.

- `tldw_chatbook/Widgets/Console/console_rail_handle.py`
  - Reusable Textual widget for collapsed rail handles. Renders a focusable button plus optional badge with fixed narrow sizing.

- `Tests/Chat/test_console_rail_state.py`
  - Pure tests for defaults, persistence keys, invalid stored data, badge priority, no-auto-open behavior, and compact responsive override.

- `Tests/UI/test_console_persistent_rails.py`
  - Mounted Console tests for first-start layout, toggles, persistence, badges, width changes, composer span, compact behavior, keyboard reachability, and existing provider recovery visibility.

### Modified Files

- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Import rail state helpers and handle widget.
  - Build rail state from workspace/session/conversation context.
  - Compose left rail, left handle, main lane, right rail, and right handle as separate named helpers.
  - Add rail-level collapse/open button handlers.
  - Persist rail preference updates and copy temporary-session preference to durable conversation key.
  - Keep the composer outside the workspace grid and do not remount it during rail toggles.

- `tldw_chatbook/Widgets/Console/__init__.py`
  - Export `ConsoleRailHandle`.

- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add handle, badge, visible rail header, and collapse-button styles.
  - Update Console width/display classes so hidden rails do not reserve space.

- `Tests/UI/test_console_internals_decomposition.py`
  - Update existing inspector/layout tests to account for right-collapsed-by-default behavior.
  - Open the Inspector explicitly in tests that inspect Inspector internals.

- `Tests/UI/test_console_native_transcript.py`
  - Update focus-order expectations to include collapsed rail handles when visible.

## Rail Contract

Implement the new pure contract in `tldw_chatbook/Chat/console_rail_state.py` before touching UI code.

Required public shape:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CONSOLE_RAIL_LEFT_DEFAULT_OPEN = True
CONSOLE_RAIL_RIGHT_DEFAULT_OPEN = False
CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS = 150
CONSOLE_RAIL_CONTEXT_LABEL = "Context"
CONSOLE_RAIL_INSPECTOR_LABEL = "Inspector"


@dataclass(frozen=True)
class ConsoleRailPreferences:
    left_open: bool = CONSOLE_RAIL_LEFT_DEFAULT_OPEN
    right_open: bool = CONSOLE_RAIL_RIGHT_DEFAULT_OPEN


@dataclass(frozen=True)
class ConsoleRailPreferenceKey:
    workspace_id: str
    scope_id: str
    value: str
    fallback_value: str | None = None


@dataclass(frozen=True)
class ConsoleRailState:
    left_open: bool
    right_open: bool
    preferred_left_open: bool
    preferred_right_open: bool
    left_label: str = CONSOLE_RAIL_CONTEXT_LABEL
    right_label: str = CONSOLE_RAIL_INSPECTOR_LABEL
    left_badge: str = ""
    right_badge: str = ""
    persistence_key: str = ""
    right_forced_collapsed: bool = False
```

Required functions:

```python
def build_console_rail_preference_key(
    *,
    workspace_id: Any = None,
    conversation_id: Any = None,
    session_id: Any = None,
) -> ConsoleRailPreferenceKey: ...


def coerce_console_rail_preferences(raw: Any) -> ConsoleRailPreferences: ...


def serialize_console_rail_preferences(
    preferences: ConsoleRailPreferences,
) -> dict[str, bool]: ...


def build_console_context_rail_badge(
    *,
    staged_source_count: Any = 0,
    staged_summary: Any = "",
    workspace_label: Any = "",
    session_label: Any = "",
) -> str: ...


def build_console_inspector_rail_badge(
    *,
    run_status: Any = None,
    inspector_rows: tuple[Any, ...] = (),
    tool_count: Any = 0,
    approval_count: Any = 0,
    can_save_chatbook: bool = False,
) -> str: ...


def build_console_rail_state(
    *,
    preference_key: ConsoleRailPreferenceKey,
    stored_preferences: Any = None,
    staged_source_count: Any = 0,
    staged_summary: Any = "",
    workspace_label: Any = "",
    session_label: Any = "",
    run_status: Any = None,
    inspector_rows: tuple[Any, ...] = (),
    tool_count: Any = 0,
    approval_count: Any = 0,
    can_save_chatbook: bool = False,
    available_columns: int | None = None,
) -> ConsoleRailState: ...
```

Badge rules:

- Context badge priority:
  - `"<n> staged"` when `staged_source_count > 0`.
  - `"staged"` when `staged_summary` describes an active staged item but count is unavailable.
  - `"workspace"` when the workspace label is specific and not the local/default/no-workspace fallback.
  - `"session"` when a session/conversation label is available but no staged or workspace badge applies.
  - `""` otherwise.
- Inspector badge priority:
  - `"failed"` when `run_status == "failed"`.
  - `"blocked"` when `run_status == "blocked"` or any Provider/RAG/source row is blocked.
  - `"<n> approval"` or `"<n> approvals"` when approvals are pending.
  - `"tools"` when tools are ready.
  - `"artifact"` when `can_save_chatbook` is true.
  - `"source"` when source/readiness rows indicate staged or ready source context.
  - `""` otherwise.

Persistence key rules:

- Use `console_rail_state:<workspace_id>:<scope_id>`.
- Prefer `workspace_id + conversation_id`, then `workspace_id + session_id`, then `workspace_id + global`, then `global:global`.
- Sanitize key parts with a deterministic allowlist of letters, numbers, `_`, `.`, and `-`; replace other runs with `_`.
- When both conversation and session ids are present, return the conversation key as `value` and the temporary session key as `fallback_value`.

Compact override:

- If `available_columns` is below `CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS`, return `right_open=False` and `right_forced_collapsed=True`.
- Preserve `preferred_right_open` from stored preferences even when the effective `right_open` is false.

## Shared Verification Commands

Use the local virtualenv when available.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py --tb=short
```

Expected result:

```text
... passed
```

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected result:

```text
... passed
```

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected result:

```text
... passed
```

```bash
git diff --check
```

Expected result: no output.

Run `git diff --check` before every implementation commit.

## Task 1: Add Pure Rail State Tests

**Files:**
- Create: `Tests/Chat/test_console_rail_state.py`

- [ ] Add tests for the first-start default.

Required assertions:

```python
key = build_console_rail_preference_key(workspace_id="workspace-1", session_id="session-1")
state = build_console_rail_state(preference_key=key)

assert state.left_open is True
assert state.right_open is False
assert state.preferred_left_open is True
assert state.preferred_right_open is False
assert state.persistence_key == "console_rail_state:workspace-1:session-1"
```

- [ ] Add tests for stored preferences restoring both booleans.

Required assertions:

```python
state = build_console_rail_state(
    preference_key=key,
    stored_preferences={"left_open": False, "right_open": True},
    available_columns=220,
)
assert state.left_open is False
assert state.right_open is True
```

- [ ] Add tests for invalid stored preferences falling back to defaults.

Use invalid examples such as `None`, `"bad"`, `{"left_open": "bad"}`, and `{"right_open": []}`.

- [ ] Add tests for deterministic persistence-key order.

Required cases:

```python
assert build_console_rail_preference_key(
    workspace_id="workspace 1",
    conversation_id="conv:1",
    session_id="session:1",
).value == "console_rail_state:workspace_1:conv_1"

assert build_console_rail_preference_key(
    workspace_id="workspace 1",
    conversation_id="conv:1",
    session_id="session:1",
).fallback_value == "console_rail_state:workspace_1:session_1"
```

- [ ] Add tests for Context badge priority.

Cover staged source count over workspace badge, active staged summary fallback, workspace badge, and empty badge.

- [ ] Add tests for Inspector badge priority.

Cover failed over blocked, blocked over approval, approval over tools, tools over artifact/source, artifact/source over empty.

- [ ] Add tests proving badge state does not mutate open booleans.

Set `stored_preferences={"left_open": False, "right_open": False}` with staged and blocked states. Assert both effective rails stay closed while badges update.

- [ ] Add tests for compact responsive override.

Required assertions:

```python
state = build_console_rail_state(
    preference_key=key,
    stored_preferences={"left_open": True, "right_open": True},
    available_columns=120,
)
assert state.left_open is True
assert state.right_open is False
assert state.preferred_right_open is True
assert state.right_forced_collapsed is True
```

- [ ] Run the focused test and confirm it fails for missing implementation.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py --tb=short
```

Expected result: import/test failures for missing `console_rail_state`.

## Task 2: Implement Pure Rail State

**Files:**
- Create: `tldw_chatbook/Chat/console_rail_state.py`

- [ ] Implement the dataclasses and constants from the Rail Contract section.
- [ ] Implement safe key-part normalization.
- [ ] Implement `build_console_rail_preference_key`.
- [ ] Implement strict boolean coercion for `coerce_console_rail_preferences`.

Accepted booleans:

- Real `bool` values.
- String values `"true"`, `"yes"`, `"1"`, `"on"`, `"false"`, `"no"`, `"0"`, `"off"` if existing config data already stores strings.

Invalid values fall back per-field to the default.

- [ ] Implement `serialize_console_rail_preferences`.
- [ ] Implement Context and Inspector badge builders.
- [ ] Implement compact-width effective state in `build_console_rail_state`.
- [ ] Run the pure tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py --tb=short
```

Expected result: all tests in `Tests/Chat/test_console_rail_state.py` pass.

- [ ] Run whitespace check.

```bash
git diff --check
```

- [ ] Commit after tests pass.

Suggested commit message:

```text
Add pure Console rail state contract
```

## Task 3: Add Rail Handle Widget Tests

**Files:**
- Create: `Tests/UI/test_console_persistent_rails.py`

- [ ] Add a mounted test that first-start Console renders the left rail and right handle.

Required assertions:

- `#console-left-rail` is visible.
- `#console-staged-context-tray` is visible.
- `#console-workspace-context` is visible.
- `#console-right-rail` is not visible or has `display=False`.
- `#console-run-inspector-state` is not visible.
- `#console-live-work-source-readiness` is not visible.
- `#console-inspector-rail-handle` is visible.
- Visible text includes `Inspector`.

- [ ] Add a mounted test that the right handle is focusable/reachable.

Use `pilot.press("tab")` until the focused element id is `console-inspector-rail-open`, with a bounded loop.

- [ ] Run the focused test and confirm it fails for missing widget/wiring.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected result: selector failures for missing handle.

## Task 4: Implement ConsoleRailHandle

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_rail_handle.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`

- [ ] Implement `ConsoleRailHandle`.

Required widget shape:

```python
class ConsoleRailHandle(Vertical):
    def __init__(
        self,
        *,
        label: str,
        badge: str = "",
        button_id: str,
        badge_id: str,
        side: str,
        **kwargs: Any,
    ) -> None: ...
```

Composition rules:

- Yield a `Button(label, id=button_id, compact=True)`.
- Yield `Static(badge, id=badge_id, markup=False)` only when `badge` is non-empty.
- Add side-specific classes: `console-rail-handle-left` or `console-rail-handle-right`.
- Set button tooltip to `Open Context rail` or `Open Inspector rail`.

- [ ] Export `ConsoleRailHandle` from `tldw_chatbook/Widgets/Console/__init__.py`.

- [ ] Add TCSS classes.

Required style intent:

```tcss
.console-rail-handle {
    width: 10;
    min-width: 10;
    height: 100%;
    min-height: 20;
    padding: 0;
    border: solid $ds-grid-line;
    background: $ds-surface-panel;
}

.console-rail-handle-button,
.console-rail-collapse-button {
    height: 1;
    min-height: 1;
}

.console-rail-handle-badge {
    height: auto;
    min-height: 1;
    width: 100%;
    text-align: center;
    color: $ds-text-primary;
    background: $ds-surface-raised;
}

.console-rail-header {
    height: 1;
    min-height: 1;
    layout: horizontal;
}
```

Keep handle width at 10 columns so `left rail + center minimum + right handle` can fit at 100 columns.

- [ ] Run a tiny import check through the focused mounted test.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected result: tests still fail until ChatScreen wiring exists, but no import error for `ConsoleRailHandle`.

## Task 5: Wire First-Start Layout Without Toggle Persistence

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`

- [ ] Import the new rail helpers and widget.

Imports needed in `chat_screen.py`:

```python
from ...Chat.console_rail_state import (
    ConsoleRailPreferences,
    ConsoleRailState,
    build_console_rail_preference_key,
    build_console_rail_state,
    coerce_console_rail_preferences,
    serialize_console_rail_preferences,
)
from ...Widgets.Console import ConsoleRailHandle
```

- [ ] Add `ChatScreen` helpers for config access.

Required helpers:

```python
def _console_config(self) -> dict[str, Any]: ...
def _console_rail_state_config(self) -> dict[str, Any]: ...
def _stored_console_rail_preferences(self, key: str, fallback_key: str | None) -> Any: ...
def _console_rail_available_columns(self) -> int | None: ...
def _current_console_run_status_value(self) -> str: ...
def _build_console_rail_state(... ) -> ConsoleRailState: ...
```

Implementation notes:

- Use `self.size.width` for available columns when mounted; return `None` during early compose if size is not available.
- `_current_console_run_status_value` should read `self._console_chat_controller.run_state.status` when the controller exists, otherwise support a test seam such as `app_instance.console_run_status_override`, otherwise return `"idle"`.

- [ ] Add compose helpers.

Suggested helper split:

```python
def _compose_console_left_rail(..., rail_state: ConsoleRailState) -> ComposeResult: ...
def _compose_console_context_handle(self, rail_state: ConsoleRailState) -> ConsoleRailHandle: ...
def _compose_console_main_column(...) -> ComposeResult: ...
def _compose_console_right_rail(..., rail_state: ConsoleRailState) -> ComposeResult: ...
def _compose_console_inspector_handle(self, rail_state: ConsoleRailState) -> ConsoleRailHandle: ...
```

- [ ] Update `compose_content()` to build `rail_state` after `control_state`, `staged_context_state`, `inspector_state`, and `workspace_context_state`.

Inputs:

- `workspace_id` from `_current_console_workspace_context().active_workspace_id`.
- `conversation_id` from `_current_console_conversation_id()`.
- `session_id` from `self._console_chat_store.active_session_id` only if the store already exists.
- `stored_preferences` from `_stored_console_rail_preferences`.
- `staged_source_count` from pending launch/control state.
- `staged_summary` from `staged_context_state.summary`.
- `workspace_label` from `workspace_context_state.workspace_label`.
- `session_label` from active session/conversation metadata when available.
- `run_status` from `_current_console_run_status_value()`.
- `inspector_rows` from `inspector_state.rows`.
- `tool_count` and `approval_count` from existing helper methods.
- `can_save_chatbook` from `inspector_state.can_save_chatbook`.

- [ ] Mount both rail panels and both handles in `#console-workspace-grid`, but hide the inactive ones with `styles.display = "none"` at composition time.

Do not remount or recreate `ConsoleComposerBar` during rail toggles.

- [ ] Add visible rail-level collapse buttons.

Left rail:

- A rail header at the top with `Static("Context")` and `Button("Hide", id="console-context-rail-collapse")`.
- Then existing `ConsoleStagedContextTray` and `ConsoleWorkspaceContextTray`.

Right rail:

- A dedicated full-rail container with id `#console-right-rail`.
- A rail header at the top with `Static("Inspector")` and `Button("Hide", id="console-inspector-rail-collapse")`.
- Then existing `ConsoleRunInspector` and source/live-work readiness content mounted inside `#console-right-rail`.

- [ ] Update CSS so:

- `#console-left-rail` keeps min width 36 when visible.
- `#console-right-rail` keeps min width 40 when visible.
- `#console-run-inspector-state` remains the inner inspector widget and must not be treated as the full rail container.
- `#console-main-column` keeps min width 52.
- Handle containers use fixed width 10.
- Hidden rails/handles do not reserve width.

- [ ] Run the first-start mounted tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected result: first-start and handle visibility tests pass; toggle/persistence tests are not added yet.

## Task 6: Add Toggle And Persistence Tests

**Files:**
- Modify: `Tests/UI/test_console_persistent_rails.py`

- [ ] Add test: clicking `#console-context-rail-collapse` hides left rail and shows `#console-context-rail-handle`.

Required assertions:

- `#console-staged-context-tray` and `#console-workspace-context` are hidden or their parent left rail is hidden.
- `#console-context-rail-handle` is visible.
- `#console-main-column.region.width` increases compared with first-start width.

- [ ] Add test: clicking `#console-inspector-rail-open` restores the right rail and hides right handle.

Required assertions:

- `#console-right-rail` visible.
- `#console-run-inspector-state` visible.
- `#console-live-work-source-readiness` or live-work status content visible when present.
- `#console-inspector-rail-handle` hidden.
- `#console-main-column.region.width` decreases compared with the right-collapsed width.

- [ ] Add test: rail state persists by workspace/session key.

Set a fake app config:

```python
app.app_config = {"console": {"rail_state": {}}}
```

Click to collapse left and open right. Assert:

```python
saved = app.app_config["console"]["rail_state"]
assert saved["console_rail_state:global:global"] == {
    "left_open": False,
    "right_open": True,
}
```

Then mount a new `ConsoleHarness(app)` and assert left is collapsed and right is open.

- [ ] Add test: workspace/session changes use a different key.

Use a fake `workspace_registry_service` returning workspace ids/names for two workspaces, or set the lowest-risk existing seam available in current tests. Assert a saved preference for workspace A does not affect workspace B.

- [ ] Add test: session preference copies to durable conversation key.

Set:

```python
app.app_config = {
    "console": {
        "rail_state": {
            "console_rail_state:global:session-1": {
                "left_open": False,
                "right_open": True,
            }
        }
    }
}
```

Set the active Console session id to `session-1`, then expose conversation id `conv-1` through `ChatScreen._current_console_conversation_id` using monkeypatch or an existing session data seam. Assert after building/syncing rail state:

```python
assert app.app_config["console"]["rail_state"]["console_rail_state:global:conv-1"] == {
    "left_open": False,
    "right_open": True,
}
```

- [ ] Run the focused tests and confirm failures before implementation.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected result: toggle/persistence failures.

## Task 7: Implement Toggle And Persistence Wiring

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [ ] Add button handlers.

Required handlers:

```python
@on(Button.Pressed, "#console-context-rail-collapse")
async def handle_console_context_rail_collapse(self, event: Button.Pressed) -> None: ...

@on(Button.Pressed, "#console-context-rail-open")
async def handle_console_context_rail_open(self, event: Button.Pressed) -> None: ...

@on(Button.Pressed, "#console-inspector-rail-collapse")
async def handle_console_inspector_rail_collapse(self, event: Button.Pressed) -> None: ...

@on(Button.Pressed, "#console-inspector-rail-open")
async def handle_console_inspector_rail_open(self, event: Button.Pressed) -> None: ...
```

- [ ] Add `_set_console_rail_preference(left_open: bool | None = None, right_open: bool | None = None)`.

Rules:

- Read the current `ConsoleRailPreferences` from config.
- Apply only the requested side change.
- Update `self.app_instance.app_config["console"]["rail_state"][key]` immediately.
- Best-effort persist to disk with `save_setting_to_cli_config("console.rail_state", key, serialized_preferences)`.
- If persistence fails, keep the in-memory preference and notify with warning severity only if this happened from user action.

- [ ] Add temporary-to-durable migration.

Rules:

- When `build_console_rail_preference_key` returns both `value` and `fallback_value`, and `value` is missing while `fallback_value` exists, copy fallback preferences into `value`.
- Update in-memory config and best-effort save.
- Do not delete the fallback key.

- [ ] Add `_sync_console_rail_visibility(rail_state: ConsoleRailState)`.

This method should update existing mounted widgets instead of recomposing the whole screen:

- Set `display`/`styles.display` for `#console-left-rail`, `#console-context-rail-handle`, `#console-right-rail`, and `#console-inspector-rail-handle`.
- Do not toggle only `#console-run-inspector-state`; it is an inner widget. Collapsing the Inspector rail must hide the rail header, inspector state, and source/live-work readiness content together.
- Update handle badge renderables if state changes.
- Leave `#console-native-composer` mounted.
- Leave `#console-session-surface` mounted.

If a dynamic mount edge makes handle badge updates awkward, call `refresh(recompose=True)` only on `ConsoleRailHandle`, not on the whole screen.

- [ ] After each toggle, rebuild rail state and call `_sync_console_rail_visibility`.

- [ ] Run the focused tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected result: toggle and persistence tests pass.

- [ ] Run pure tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py --tb=short
```

- [ ] Run `git diff --check`.
- [ ] Commit after tests pass.

Suggested commit message:

```text
Wire Console rail toggles and persistence
```

## Task 8: Add Badge And No-Auto-Open Mounted Tests

**Files:**
- Modify: `Tests/UI/test_console_persistent_rails.py`

- [ ] Add provider blocked badge test.

Setup:

```python
app.app_config = {
    "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
    "api_settings": {"openai": {"api_key": ""}},
    "console": {"rail_state": {"console_rail_state:global:global": {"left_open": True, "right_open": False}}},
}
app.chat_api_provider_value = "OpenAI"
app.chat_api_model_value = "gpt-4.1-2025-04-14"
```

Assert:

- Right Inspector remains collapsed.
- `#console-inspector-rail-badge` includes `blocked`.
- Central `#console-provider-recovery-strip` is visible.
- `#console-open-provider-settings` is visible and enabled.

- [ ] Add failed-over-blocked priority test.

Set `app.console_run_status_override = "failed"` and provider blocked. Assert the right handle badge is `failed`, not `blocked`.

- [ ] Add pending approval badge test.

Set `app.console_pending_approval_count = 1` while right rail collapsed. Assert:

- Right handle badge includes `1 approval`.
- Right rail remains collapsed.

- [ ] Add tool badge test.

Set `app.console_tool_count = 2`. Assert badge is `tools` when no failed/blocked/approval state exists.

- [ ] Add left staged-context badge test.

Provide `pending_console_launch = ConsoleLiveWorkLaunch.from_values(...)`, save left preference collapsed, and assert:

- Left rail remains collapsed.
- Left handle badge is `1 staged` or `staged`.

- [ ] Add no-auto-open test for state updates after mount.

Start with right collapsed and no approval, then set `app.console_pending_approval_count = 1`, call the existing sync method or newly added rail sync method, pause, and assert the Inspector remains collapsed while badge updates.

- [ ] Run the focused tests and confirm failures before implementation if the badge sync does not yet exist.

## Task 9: Implement Badge Sync

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py` if needed

- [ ] Add `ConsoleRailHandle.sync_state(label: str, badge: str)`.

Use `refresh(recompose=True)` on the handle only. Keep this local to the handle and avoid screen-wide recompose.

- [ ] Update `_sync_console_control_bar()` or the closest existing central state refresh seam to rebuild rail state and call `_sync_console_rail_visibility`.

Important existing sync paths:

- Provider/model changes call `_sync_console_control_bar()`.
- Native send/stop transitions call `_refresh_console_native_run_state()`.
- Native session sync calls `_sync_native_console_chat_ui()`.

The implementation should update badges from those existing refresh paths without introducing a timer.

- [ ] Ensure no state path calls an auto-open method.

Search for rail open calls after implementation:

```bash
rg -n "console_.*rail.*open|right_open|left_open|Inspector.*open|Context.*open" tldw_chatbook/UI/Screens/chat_screen.py
```

Only user event handlers and preference/state builders should change rail booleans.

- [ ] Run badge/no-auto-open tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected result: all persistent-rails mounted tests pass.

## Task 10: Protect Compact Widths And Composer Span

**Files:**
- Modify: `Tests/UI/test_console_persistent_rails.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`

- [ ] Add compact-width tests at 100, 120, and 140 columns.

Required assertions for each:

- `#console-main-column.region.width >= 52`.
- `#console-native-composer.region.width >= #console-workspace-grid.region.width - 2`.
- Right Inspector is collapsed at 100/120 even if stored preference says `right_open=True`.
- `app.app_config["console"]["rail_state"][key]["right_open"]` remains `True` after compact rendering.
- Handle visible text fits the handle region width.

- [ ] Add desktop-width composer span test.

At 212 columns:

- First-start: composer spans at least the workspace grid width minus frame/padding.
- Left collapsed: composer width is unchanged while main column width increases.
- Right open: composer width is unchanged while main column width decreases.

- [ ] Update CSS or layout widths until the tests pass.

Implementation guidance:

- Keep `#console-native-composer` outside `#console-workspace-grid`.
- Do not put the composer inside `#console-main-column`.
- Keep handles at fixed width 10.
- Avoid adding margins that make the composer narrower than the workspace grid.

- [ ] Run focused layout tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

## Task 11: Update Existing Console Tests For New Default

**Files:**
- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify only if needed: `Tests/UI/test_console_workspace_context_rail.py`, `Tests/UI/test_console_native_chat_flow.py`

- [ ] Update tests that currently assume `#console-run-inspector` is visible on first start.

For tests about Inspector internals, open the Inspector explicitly:

```python
await pilot.click("#console-inspector-rail-open")
await pilot.pause()
await _wait_for_selector(console, pilot, "#console-run-inspector-state")
```

If `pilot.click(selector)` is not available in the local helper style, query the button and call the handler directly through `Button.Pressed`.

- [ ] Update workbench weight assertions.

New default expected shape:

- Main column is wider than left rail.
- Main column is wider than the collapsed Inspector handle.
- Right handle is visible.
- Right Inspector is hidden.

Add a separate both-open assertion after opening the Inspector to preserve power-user layout coverage.

- [ ] Update frame assertions.

First-start frames should include:

- `#console-workspace-grid`
- `#console-left-rail`
- `#console-staged-context-tray`
- `#console-workspace-context`
- `#console-transcript-region`
- `#console-inspector-rail-handle`
- `#console-native-composer`

Both-open tests should assert `#console-right-rail` frame and then inspect `#console-run-inspector-state` inside it.

- [ ] Update focus-order tests.

Collapsed handle buttons are expected focus targets. Do not assert that tab reaches only transcript/composer.

- [ ] Run the regression group.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected result: all pass.

- [ ] Run `git diff --check`.
- [ ] Commit after tests pass.

Suggested commit message:

```text
Update Console tests for persistent rails
```

## Task 12: Run Combined Console Verification

**Files:**
- No code changes expected unless verification exposes a defect.

- [ ] Run pure rail and display-state tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py Tests/Chat/test_console_display_state.py --tb=short
```

- [ ] Run persistent rail mounted tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

- [ ] Run Console regression tests.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

- [ ] Run a broader UI smoke only if the focused suites pass.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py --tb=short
```

- [ ] Run `git diff --check`.

Expected result: all commands pass and whitespace check is clean.

## Task 13: Capture Actual Screenshot QA

**Files:**
- Create: `Docs/superpowers/qa/console-persistent-rails/2026-05-24-console-persistent-rails.md`
- Create PNG screenshots under: `Docs/superpowers/qa/console-persistent-rails/`

- [ ] Start the app through the repo textual-web/CDP workflow.

Use `Docs/superpowers/qa/product-maturity/screen-qa/textual-web-cdp-debugging.md`. Do not use generated mockups, ASCII, or geometry dumps as approval evidence.

Server command template:

```bash
WORKTREE=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
SERVE=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve
PORT=8874
QA_HOME=/private/tmp/tldw-chatbook-console-rails-home
QA_CONFIG=/private/tmp/tldw-chatbook-console-rails-config
QA_DATA=/private/tmp/tldw-chatbook-console-rails-data
mkdir -p "$QA_HOME" "$QA_CONFIG" "$QA_DATA"
PYTHONPATH="$WORKTREE" HOME="$QA_HOME" XDG_CONFIG_HOME="$QA_CONFIG" XDG_DATA_HOME="$QA_DATA" "$SERVE" --host 127.0.0.1 --port "$PORT"
```

If local port binding is blocked by the sandbox, rerun this command with escalation. Do not accept SVG-only evidence.

Initial temp config after first config creation:

```toml
[general]
default_tab = "chat"

[splash_screen]
enabled = false
```

Playwright capture template:

```python
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path("Docs/superpowers/qa/console-persistent-rails/2026-05-24-first-start-left-open-right-collapsed.png").resolve()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 2050, "height": 1240}, device_scale_factor=1)
    page.goto("http://127.0.0.1:8874", wait_until="domcontentloaded")
    page.wait_for_timeout(7000)
    page.evaluate(
        """
        document.body.classList.add("-first-byte");
        for (const el of document.querySelectorAll(".intro-dialog,.closed-dialog,.shade")) {
          el.style.pointerEvents = "none";
        }
        """
    )
    page.screenshot(path=str(out), full_page=True)
    browser.close()
    print(out)
```

For the remaining states, use mounted tests or temporary config setup to put Console into the desired rail/provider/approval/staged state, then capture through the same textual-web/CDP path.

- [ ] Capture PNG: first-start left open/right collapsed.

Suggested file:

```text
Docs/superpowers/qa/console-persistent-rails/2026-05-24-first-start-left-open-right-collapsed.png
```

- [ ] Capture PNG: provider blocked with right collapsed badge and central `Open Settings` recovery visible.

- [ ] Capture PNG: pending approval badge with right collapsed.

- [ ] Capture PNG: failed-run badge with right collapsed.

- [ ] Capture PNG: staged-context badge with left collapsed.

- [ ] Capture PNG: both rails open.

- [ ] Capture PNG: both rails collapsed.

- [ ] Verify each file is a PNG.

```bash
file Docs/superpowers/qa/console-persistent-rails/*.png
```

Expected result: every file reports `PNG image data`.

- [ ] Inspect screenshots before presenting them.

Reject and recapture if any screenshot shows:

- Loader/blank screen.
- Wrong route.
- Overlapping text.
- Hidden composer.
- Missing central provider recovery in provider-blocked state.
- Handle badge text clipped beyond recognition.
- Right rail auto-opened because of badge state.

- [ ] Write the QA note.

Required note sections:

- Scope.
- Screenshots.
- Verification commands.
- UX findings.
- Residual risks.
- User approval status.

- [ ] Ask the user for explicit screenshot approval before claiming the UI is visually approved.

## Task 14: Final Verification And Handoff

**Files:**
- No code changes expected unless final verification exposes a defect.

- [ ] Run focused tests one final time.

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py Tests/Chat/test_console_display_state.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

- [ ] Run `git diff --check`.

- [ ] Review the diff manually.

Checklist:

- No production behavior outside Console rails changed unintentionally.
- No screen-wide recompose on rail toggles that can drop drafts.
- No auto-open behavior.
- Persisted config stores only booleans.
- Compact override does not mutate stored preferences.
- Existing provider recovery and Settings route remain central and actionable.
- Screenshot QA note exists and references actual PNG paths.

- [ ] Commit final fixes if needed.

Suggested final commit message if a final cleanup commit is needed:

```text
Polish Console persistent rail QA
```

## Completion Criteria

- [ ] First-start Console shows left Context open and right Inspector collapsed.
- [ ] Context and Inspector can each be collapsed/reopened through visible in-layout handles.
- [ ] Rail booleans persist per workspace/session, with temporary session preference copied to durable conversation key.
- [ ] Collapsed badges are deterministic and never auto-open rails.
- [ ] Compact 100/120/140-column layouts keep the center lane readable and preserve stored right-open preference.
- [ ] Composer remains full-width under the outer Console workbench across rail states.
- [ ] Existing Console provider recovery, send-blocked draft preservation, transcript selection, tab controls, staged context, workspace context, and Inspector behavior continue to pass.
- [ ] Focused pure and mounted tests pass.
- [ ] Actual screenshot QA exists for required states and has user approval before visual approval is claimed.
