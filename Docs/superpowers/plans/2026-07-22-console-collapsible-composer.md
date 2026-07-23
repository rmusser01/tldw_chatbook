# Collapsible Console Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Console users manually collapse the five-to-eight-row composer into an exact one-row restore bar while preserving unsent work, run control, keyboard accessibility, and transcript reading position.

**Architecture:** `ChatScreen` owns one transient Console-wide collapsed boolean and orchestrates focus plus transcript-position restoration. The existing mounted `ConsoleComposerBar` gains two stable child presentations and remains the sole owner of draft/editor/action state. A state-aware focus resolver keeps the existing F6 pane registry intact while selecting either the expanded composer root or collapsed Expand button.

**Spec:** `Docs/superpowers/specs/2026-07-22-console-collapsible-composer-design.md` — read it before starting.

**Backlog:** `TASK-398` — `backlog/tasks/task-398 - Add-collapsible-Console-composer-reading-mode.md`.

**ADR required:** no

**ADR path:** `backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** This is focused Console UI behavior implemented within ADR-011's existing stable compose tree, widget-message, and screen-owned orchestration boundaries. It changes no storage, schema, service, security, dependency, or cross-module contract.

**Tech Stack:** Python ≥3.11, Textual 8.x, Rich, TCSS, pytest/pytest-asyncio, Textual `App.run_test()`, textual-serve, Playwright.

## Global Constraints

- Work only in the current repository checkout and preserve unrelated user changes in the dirty worktree.
- Do not create a second persisted preference. `_console_composer_collapsed` is screen-instance memory only.
- Do not remount the composer during a normal collapse/expand transition.
- Do not serialize caret, selection, paste-token display state, or collapse state to config/database/session records.
- Do not edit `tldw_chatbook/css/tldw_cli_modular.tcss` by hand. Edit `_agentic_terminal.tcss`, then regenerate the bundle with `build_css.py`.
- Existing selector ids for draft and actions remain unchanged.
- The separate top-area status strip is out of scope; it has not landed in this checkout. The new composer layout must not depend on it.
- Keep the existing non-priority expanded-mode Escape binding. Add a separate dynamically enabled priority Escape binding only for collapsed mode.
- Hidden composer input is inert: printable keys, editing keys, Enter, paste, and dropped paths must never mutate or send a collapsed draft.
- Setup-modal blocking remains authoritative over every new button and binding.
- Each task starts with a failing test, makes the smallest implementation change, reruns the focused tests, and commits only its own files.

## File Map

- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  - Add stable expanded/collapsed presentations, collapsed status derivation, one-row geometry, and cursor-timer gating.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  - Add screen-owned state, dynamic Escape action, state-aware focus targets, transitions, input guards, transcript-position restore, setup/run integration, and initial state propagation.
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Style the two presentations and exact one-row collapsed geometry.
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_console_composer_collapse.py`
  - Focused widget, interaction, lifecycle, scroll, and responsive-layout coverage.
- Modify: `Tests/UI/test_console_command_composer.py`
  - Regression for clearing the unknown-command literal-send arm.
- Modify: `Tests/UI/test_workbench_pane_focus.py`
  - Regression for state-aware F6 focus resolution.
- Modify: `Tests/UI/test_console_native_chat_flow.py`
  - Reuse the real stop/setup/tab/navigation harnesses for integration coverage.
- Create: `Docs/superpowers/qa/console-collapsible-composer-2026-07/README.md`
  - Record live Textual-web capture recipe, evidence, and approval status.
- Create during QA: PNG evidence in `Docs/superpowers/qa/console-collapsible-composer-2026-07/`.
- Modify: `backlog/tasks/task-398 - Add-collapsible-Console-composer-reading-mode.md`
  - Check acceptance criteria, add implementation notes, and mark Done only after tests, static checks, screenshot evidence, and user approval.

---

### Task 1: Add the stable dual-presentation composer widget

**Files:**

- Create: `Tests/UI/test_console_composer_collapse.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`

**Interfaces:**

- `ConsoleComposerBar(*, collapsed: bool = False, collapse_large_pastes: bool = True, paste_collapse_threshold: int = DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD, **kwargs: Any)`
- `ConsoleComposerBar.collapsed -> bool`
- `ConsoleComposerBar.set_collapsed(collapsed: bool) -> None`
- Stable selectors:
  - `#console-composer-expanded`
  - `#console-composer-collapse`
  - `#console-composer-collapsed`
  - `#console-composer-collapsed-status`
  - `#console-collapsed-stop-generation`
  - `#console-composer-expand`

- [ ] **Step 1: Create focused mounted-test helpers**

Create `Tests/UI/test_console_composer_collapse.py` with the common ready-Console setup:

```python
"""Mounted regressions for the collapsible Console composer."""

import pytest
from textual.events import Paste
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_console_native_chat_flow import (
    WaitingGateway,
    _configure_native_ready_console,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript


def _ready_console_host() -> ConsoleHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


async def _mounted_console(host: ConsoleHarness, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    return console
```

- [ ] **Step 2: Write failing default/geometry/idempotency tests**

Add:

```python
@pytest.mark.asyncio
async def test_console_composer_defaults_expanded_and_collapses_to_exactly_one_row():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        expanded = composer.query_one("#console-composer-expanded")
        collapsed = composer.query_one("#console-composer-collapsed")

        assert composer.collapsed is False
        assert expanded.display is True
        assert collapsed.display is False
        assert 5 <= composer.region.height <= 8

        composer.set_collapsed(True)
        composer.set_collapsed(True)
        await pilot.pause()

        assert composer.collapsed is True
        assert expanded.display is False
        assert collapsed.display is True
        assert composer.region.height == 1
        assert composer.can_focus is False

        composer.set_collapsed(False)
        composer.set_collapsed(False)
        await pilot.pause()

        assert composer.collapsed is False
        assert 5 <= composer.region.height <= 8
        assert composer.can_focus is True
```

Add a second test at `size=(100, 32)` asserting:

```python
assert composer.region.height == 1
assert composer.query_one("#console-composer-expand", Button).region.width > 0
assert composer.query_one("#console-composer-collapsed-status", Static).region.width > 0
```

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_composer_collapse.py \
  -k "defaults_expanded or compact" --tb=short
```

Expected: FAIL because the new constructor argument, property, containers, and controls do not exist.

- [ ] **Step 3: Write failing preservation/status/run-state tests**

Add parameterized status assertions:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "attachment", "run_active", "expected"),
    [
        ("", None, False, "Composer hidden"),
        (" ", None, False, "Composer hidden · Draft retained"),
        ("draft", "photo.png · 12 B", False,
         "Composer hidden · Draft retained · Attachment retained"),
        ("", None, True, "Composer hidden · Generating"),
        ("draft", "photo.png · 12 B", True,
         "Composer hidden · Generating · Draft retained · Attachment retained"),
    ],
)
async def test_console_collapsed_status_uses_presence_only(
    draft, attachment, run_active, expected
):
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft(draft)
        composer.set_pending_attachment_label(attachment)
        composer.sync_action_state(
            has_draft=bool(draft.strip()),
            run_active=run_active,
            can_save_chatbook=False,
        )
        composer.set_collapsed(True)
        await pilot.pause()

        status = composer.query_one("#console-composer-collapsed-status", Static)
        assert str(status.renderable) == expected
        stop = composer.query_one("#console-collapsed-stop-generation", Button)
        assert stop.display is run_active
        assert "photo.png" not in str(status.renderable)
```

Add a round-trip test which:

1. Calls `insert_pasted_text()` with text above the collapse threshold.
2. Moves the caret left once and records `draft_text()`, `cursor_index`, `has_paste_segments()`, and `has_full_draft_selection()`.
3. Calls `select_all_draft()` and records selection.
4. Calls `set_pending_attachment_label("photo.png · 12 B")`.
5. Calls `set_collapsed(True)` then `set_collapsed(False)`.
6. Asserts canonical text, paste provenance, caret, full-draft selection, and attachment indicator text are unchanged.

Add a pending-unfurl test using the existing paste-token click helper pattern from `test_console_internals_decomposition.py`: arm `Unfurl?`, call `set_collapsed(True)` only to prove the widget itself preserves the segment, then leave confirmation reset for the screen-transition test in Task 2.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_composer_collapse.py \
  -k "status or round_trip or geometry" --tb=short
```

Expected: FAIL on missing presentation/state APIs.

- [ ] **Step 4: Implement the stable presentation tree**

In `ConsoleComposerBar.__init__`, add the initial state without querying unmounted children:

```python
def __init__(
    self,
    *,
    collapsed: bool = False,
    collapse_large_pastes: bool = True,
    paste_collapse_threshold: int = DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    **kwargs: Any,
) -> None:
    super().__init__(**kwargs)
    self._collapsed = bool(collapsed)
    self.can_focus = not self._collapsed
    # Existing draft/action initialization follows unchanged.
```

Add:

```python
@property
def collapsed(self) -> bool:
    """Return whether the compact restore-only presentation is active."""
    return self._collapsed


def _collapsed_status_text(self) -> str:
    parts = ["Composer hidden"]
    if self._run_active:
        parts.append("Generating")
    if bool(self.draft_text()):
        parts.append("Draft retained")
    if self._pending_attachment_label is not None:
        parts.append("Attachment retained")
    return " · ".join(parts)
```

Refactor `compose()` so both containers always exist. Move every current child into `#console-composer-expanded`, replace the current title `Static` with a fixed-width button, then add the collapsed sibling:

```python
expanded = Horizontal(
    id="console-composer-expanded",
    classes="console-composer-presentation",
)
expanded.styles.display = "none" if self._collapsed else "block"
with expanded:
    yield self._bounded_button(
        "Composer ▾",
        width=10,
        id="console-composer-collapse",
        classes="destination-action-button console-composer-toggle",
        tooltip="Collapse composer for more transcript space.",
    )
    # Yield the unchanged visible draft, recovery, attachment, hidden Input,
    # hidden status/reason, and #console-composer-actions widgets here.

collapsed = Horizontal(
    id="console-composer-collapsed",
    classes="console-composer-presentation",
)
collapsed.styles.display = "block" if self._collapsed else "none"
with collapsed:
    yield Static(
        self._collapsed_status_text(),
        id="console-composer-collapsed-status",
    )
    collapsed_stop = self._bounded_button(
        "Stop",
        width=8,
        id="console-collapsed-stop-generation",
        classes="destination-action-button console-stop-button",
        variant="warning",
        tooltip="Stop generation in the active Console session.",
    )
    collapsed_stop.styles.display = "block" if self._run_active else "none"
    yield collapsed_stop
    yield self._bounded_button(
        "Expand ▴",
        width=10,
        id="console-composer-expand",
        classes="destination-action-button console-composer-toggle",
        tooltip="Expand composer and return to the draft.",
    )
```

Do not rename existing draft/action ids.

- [ ] **Step 5: Implement idempotent geometry and state synchronization**

Add focused helpers:

```python
def _apply_collapsed_geometry(self) -> None:
    self.styles.height = 1
    self.styles.min_height = 1
    self.styles.max_height = 1
    self.refresh(layout=True)


def _sync_collapsed_presentation(self) -> None:
    try:
        expanded = self.query_one("#console-composer-expanded", Horizontal)
        collapsed = self.query_one("#console-composer-collapsed", Horizontal)
        status = self.query_one("#console-composer-collapsed-status", Static)
        stop = self.query_one("#console-collapsed-stop-generation", Button)
    except NoMatches:
        return
    expanded.styles.display = "none" if self._collapsed else "block"
    collapsed.styles.display = "block" if self._collapsed else "none"
    status.update(self._collapsed_status_text())
    stop.styles.display = "block" if self._run_active else "none"
    self.set_class(self._collapsed, "console-composer-collapsed")


def set_collapsed(self, collapsed: bool) -> None:
    """Switch presentation without remounting or clearing editor state."""
    collapsed = bool(collapsed)
    self._collapsed = collapsed
    self.can_focus = not collapsed
    self._sync_collapsed_presentation()
    self._sync_cursor_blink_state()
    if collapsed:
        self._apply_collapsed_geometry()
    else:
        self._refresh_visible_draft()
```

Harden existing paths:

```python
def _refresh_visible_draft(self) -> None:
    if self._collapsed:
        self._sync_collapsed_presentation()
        self._apply_collapsed_geometry()
        return
    # Existing rendering and _apply_draft_height logic.


def _sync_cursor_blink_state(self) -> None:
    # Existing initialization.
    if timer is None:
        return
    if self.has_focus_within and not self._collapsed:
        timer.resume()
    else:
        timer.pause()
```

Call `_sync_collapsed_presentation()` after run-state and attachment-state changes. `on_mount()` must apply the initial collapsed geometry instead of calling the expanded draft-height path.

- [ ] **Step 6: Add source TCSS and regenerate the bundle**

In `_agentic_terminal.tcss`, keep expanded sizing unchanged and add:

```css
#console-composer-expanded,
#console-composer-collapsed {
    width: 1fr;
    min-width: 0;
    height: 1;
    min-height: 1;
    layout: horizontal;
    align: left middle;
}

#console-native-composer.console-composer-collapsed {
    height: 1;
    min-height: 1;
    max-height: 1;
    padding: 0;
    border: none;
    background: $ds-surface-raised;
}

#console-composer-collapse {
    width: 10;
    min-width: 10;
}

#console-composer-collapsed-status {
    width: 1fr;
    min-width: 0;
    height: 1;
    min-height: 1;
    text-overflow: ellipsis;
    text-wrap: nowrap;
    color: $ds-text-muted;
}

#console-collapsed-stop-generation {
    width: 8;
    min-width: 8;
}

#console-composer-expand {
    width: 10;
    min-width: 10;
}
```

Remove/replace the obsolete `#console-composer-title` width rule after the `Static` is replaced.

Run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
git diff --check
```

Expected: CSS build exits 0; only the source component and generated bundle change.

- [ ] **Step 7: Add a bundle-pin regression**

In `test_console_composer_collapse.py`, read both source and generated stylesheets and assert each contains:

```python
required = (
    "#console-native-composer.console-composer-collapsed",
    "#console-composer-collapsed-status",
    "#console-composer-expand",
    "text-overflow: ellipsis",
)
```

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_composer_collapse.py --tb=short
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add \
  Tests/UI/test_console_composer_collapse.py \
  tldw_chatbook/Widgets/Console/console_composer_bar.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss
git diff --cached --check
git commit -m "feat(console): add collapsible composer presentation"
```

---

### Task 2: Wire screen transitions, keyboard behavior, and safety guards

**Files:**

- Modify: `Tests/UI/test_console_composer_collapse.py`
- Modify: `Tests/UI/test_console_command_composer.py`
- Modify: `Tests/UI/test_workbench_pane_focus.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

**Interfaces:**

- `_ConsoleTranscriptReadingState`
- `ChatScreen._console_composer_collapsed: bool`
- `ChatScreen._console_composer_layout_revision: int`
- `ChatScreen.action_expand_collapsed_console_composer()`
- `ChatScreen.check_action(action, parameters)`
- `ChatScreen._console_workbench_focus_targets(pane_id)`
- `ChatScreen._set_console_composer_collapsed(collapsed)`

- [ ] **Step 1: Write failing toggle/focus/input tests**

Add mounted tests with these exact outcomes:

```python
@pytest.mark.asyncio
async def test_collapse_button_moves_focus_to_transcript_without_sending():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep me")
        store = console._ensure_console_chat_store()
        message_count = len(store.messages_for_session(store.active_session_id))
        collapse = composer.query_one("#console-composer-collapse", Button)
        collapse.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert composer.collapsed is True
        assert composer.draft_text() == "keep me"
        assert isinstance(host.focused, ConsoleTranscript)
        assert (
            len(store.messages_for_session(store.active_session_id))
            == message_count
        )


@pytest.mark.asyncio
async def test_expand_button_and_one_escape_expand_and_focus_draft():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.click("#console-composer-collapse")
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        assert composer.collapsed is False
        assert host.focused is composer

        await pilot.click("#console-composer-collapse")
        await pilot.pause()
        expand = composer.query_one("#console-composer-expand", Button)
        expand.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert composer.collapsed is False
        assert host.focused is composer
```

Add a hidden-input regression which collapses a nonempty draft, then sends:

```python
for key in ("x", "backspace", "delete", "enter"):
    await pilot.press(key)
```

and posts a `Paste("pasted")`; after each, assert the draft and transcript message count are unchanged. Also paste an attachable path-shaped string and assert no attachment worker starts.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_composer_collapse.py \
  -k "button or escape or hidden_input" --tb=short
```

Expected: FAIL because `ChatScreen` does not handle the new controls or state.

- [ ] **Step 2: Write failing F6/dynamic-Escape/setup tests**

In `Tests/UI/test_workbench_pane_focus.py`, extend the Console F6 harness:

```python
console._set_console_composer_collapsed(True)
await pilot.pause()
console.query_one("#console-inspector-rail-collapse").focus()
await pilot.press("f6")
await _wait_for_focused_id(host, pilot, "console-composer-expand")
assert console.query_one("#console-native-composer").can_focus is False
console._ensure_console_workbench_targets_focusable()
assert console.query_one("#console-native-composer").can_focus is False
```

In `test_console_composer_collapse.py`, assert:

```python
assert console.check_action("expand_collapsed_console_composer", ()) is False
console._set_console_composer_collapsed(True)
assert console.check_action("expand_collapsed_console_composer", ()) is True
```

Then monkeypatch `_console_setup_modal_blocking` to `True` and assert it becomes `False` again and neither toggle changes state.

Also select a transcript message while expanded and press Escape once. Assert
the transcript's existing non-priority binding clears selection before focus
returns to the composer. This pins the fallback behavior when the new priority
action is dynamically disabled.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_workbench_pane_focus.py \
  Tests/UI/test_console_composer_collapse.py \
  -k "collapsed or dynamic or setup" --tb=short
```

Expected: FAIL on missing screen state/action/resolver.

- [ ] **Step 3: Write the unknown-command and unfurl safety regressions**

In `Tests/UI/test_console_command_composer.py`, add a test after the existing unknown-command arm tests:

```python
@pytest.mark.asyncio
async def test_console_collapse_disarms_unknown_command_literal_send():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        assert console._console_unknown_send_armed == "/nope x"

        console._set_console_composer_collapsed(True)
        console._set_console_composer_collapsed(False)
        await pilot.pause()
        assert console._console_unknown_send_armed is None

        console.query_one("#console-send-message", Button).press()
        await pilot.pause()
        submit_spy.assert_not_called()
        assert console._console_unknown_send_armed == "/nope x"
```

Add a companion mounted test that arms an `Unfurl?` paste token, collapses through the screen action, and asserts:

```python
assert composer.has_pending_paste_confirmation() is False
assert composer.has_paste_segments() is True
assert composer.draft_text() == pasted_text
```

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_console_composer_collapse.py \
  -k "collapse_disarms or unfurl" --tb=short
```

Expected: FAIL because collapse does not yet clear either transient confirmation.

- [ ] **Step 4: Add screen-owned state and initial-state propagation**

Change the import to:

```python
from dataclasses import asdict, dataclass, replace
```

Near the Console focus constants add:

```python
@dataclass(frozen=True)
class _ConsoleTranscriptReadingState:
    anchored: bool
    scroll_y: float
    selected_message_id: str | None
```

In `ChatScreen.__init__` add:

```python
self._console_composer_collapsed = False
self._console_composer_layout_revision = 0
```

In `compose_content`, pass:

```python
ConsoleComposerBar(
    id="console-native-composer",
    classes="ds-panel",
    collapsed=self._console_composer_collapsed,
    collapse_large_pastes=self._console_collapse_large_pastes_enabled(),
    paste_collapse_threshold=self._console_paste_collapse_threshold(),
)
```

Before yielding the new composer, seed it from the already-synchronized active
session draft when available:

```python
composer = ConsoleComposerBar(
    id="console-native-composer",
    classes="ds-panel",
    collapsed=self._console_composer_collapsed,
    collapse_large_pastes=self._console_collapse_large_pastes_enabled(),
    paste_collapse_threshold=self._console_paste_collapse_threshold(),
)
store = self._console_chat_store
if store is not None and store.active_session_id is not None:
    try:
        composer.load_draft(store.session_draft(store.active_session_id))
    except KeyError:
        pass
yield self._frame_console_region(composer)
```

This closes the fallback-recompose hole where
`_console_visible_draft_session_id` still equals the active id and a later
`_sync_console_session_draft()` would otherwise take its same-session fast
path without loading the replacement widget. Do not add collapse state to
`ChatScreenState`, `ConsoleChatStore`, app config, or session settings.

- [ ] **Step 5: Add the dynamic priority Escape action**

Add before the existing non-priority Escape entry:

```python
Binding(
    "escape",
    "expand_collapsed_console_composer",
    "Composer",
    show=False,
    priority=True,
),
```

Add:

```python
def check_action(
    self,
    action: str,
    parameters: tuple[object, ...],
) -> bool | None:
    if action == "expand_collapsed_console_composer":
        return (
            self._console_composer_collapsed
            and not self._console_setup_modal_blocking()
        )
    return super().check_action(action, parameters)


def action_expand_collapsed_console_composer(self) -> None:
    """Expand the hidden Console composer and return keyboard focus to it."""
    if self._console_setup_modal_blocking():
        return
    self._set_console_composer_collapsed(False)
```

Keep `action_focus_console_composer_home()` and its non-priority binding unchanged. The local Textual probe already established that returning `False` from the priority action allows the existing non-priority chain to run.

- [ ] **Step 6: Add one state-aware focus target resolver and use it everywhere**

Add:

```python
def _console_workbench_focus_targets(self, pane_id: str) -> tuple[str, ...]:
    if pane_id == "console-native-composer":
        if self._console_composer_collapsed:
            return ("console-composer-expand",)
        return ("console-native-composer",)
    return CONSOLE_FOCUS_TARGETS_BY_PANE.get(pane_id, (pane_id,))
```

Use this helper in both:

```python
_focus_console_workbench_target()
_ensure_console_workbench_targets_focusable()
```

Update `_focus_console_composer_if_needed(force: bool = False)` so a collapsed call focuses `#console-composer-expand`, never the composer root:

```python
if self._console_composer_collapsed:
    self._focus_console_workbench_target("console-native-composer")
    return
```

Update `_apply_console_setup_block`:

```python
composer.can_focus = not blocking and not self._console_composer_collapsed
```

When setup becomes non-blocking while collapsed, `_restore_console_workbench_focus()` will resolve the composer pane to Expand.

- [ ] **Step 7: Implement the transition and stale-callback guard**

Add the reading-state helpers and transition:

```python
def _capture_console_transcript_reading_state(
    self,
) -> _ConsoleTranscriptReadingState | None:
    try:
        transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
    except QueryError:
        return None
    return _ConsoleTranscriptReadingState(
        anchored=bool(transcript.is_anchored),
        scroll_y=float(transcript.scroll_y),
        selected_message_id=transcript.selected_message_id,
    )


def _restore_console_transcript_reading_state(
    self,
    state: _ConsoleTranscriptReadingState | None,
) -> None:
    if state is None:
        return
    try:
        transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
    except QueryError:
        return
    transcript.selected_message_id = state.selected_message_id
    if state.anchored:
        transcript.anchor()
        return
    transcript.release_anchor()
    transcript.scroll_to(
        y=min(state.scroll_y, float(transcript.max_scroll_y)),
        animate=False,
    )


def _set_console_composer_collapsed(self, collapsed: bool) -> None:
    if self._console_setup_modal_blocking():
        return
    collapsed = bool(collapsed)
    composer = self._console_composer_or_none()
    if composer is None or self._console_composer_collapsed == collapsed:
        return
    reading_state = self._capture_console_transcript_reading_state()
    self._console_composer_collapsed = collapsed
    self._console_composer_layout_revision += 1
    revision = self._console_composer_layout_revision
    if collapsed:
        self._console_unknown_send_armed = None
        composer.reset_pending_unfurl()
    composer.set_collapsed(collapsed)
    self.call_after_refresh(
        self._finish_console_composer_layout_change,
        revision,
        collapsed,
        reading_state,
    )


def _finish_console_composer_layout_change(
    self,
    revision: int,
    expected_collapsed: bool,
    reading_state: _ConsoleTranscriptReadingState | None,
) -> None:
    if (
        revision != self._console_composer_layout_revision
        or expected_collapsed != self._console_composer_collapsed
    ):
        return
    self._restore_console_transcript_reading_state(reading_state)
    if expected_collapsed:
        self._focus_console_workbench_target("console-transcript-surface")
    else:
        self._focus_console_workbench_target("console-native-composer")
```

If the transcript target is missing, the helper returns safely; do not recompose.

- [ ] **Step 8: Wire button messages and stop/input guards**

Add explicit button handlers:

```python
@on(Button.Pressed, "#console-composer-collapse")
def handle_console_composer_collapse(self, event: Button.Pressed) -> None:
    event.stop()
    self._set_console_composer_collapsed(True)


@on(Button.Pressed, "#console-composer-expand")
def handle_console_composer_expand(self, event: Button.Pressed) -> None:
    event.stop()
    self._set_console_composer_collapsed(False)
```

In screen-level `on_button_pressed`, route either Stop id through the existing handler:

```python
if button_id in {
    "console-stop-generation",
    "console-collapsed-stop-generation",
}:
    await self.handle_console_stop_generation(event)
    return
```

Make the shared Stop handler setup-aware before it calls the controller:

```python
async def handle_console_stop_generation(self, event: Button.Pressed) -> None:
    event.stop()
    if self._console_setup_modal_blocking():
        return
    await self._stop_console_generation_from_visible_action()
```

Harden `_should_capture_console_input`:

```python
def _should_capture_console_input(self, composer: ConsoleComposerBar) -> bool:
    if composer.collapsed:
        return False
    focused = self.app.focused
    if getattr(focused, "id", None) in {
        "console-composer-collapse",
        "console-composer-expand",
        "console-collapsed-stop-generation",
    }:
        return False
    if focused is None:
        return True
    return self._is_descendant_or_self(
        focused, composer
    ) or self._is_legacy_chat_input_focus(focused)
```

The collapsed check protects `on_key`, `on_paste`, and drag/drop because both routes already call this helper.

Also short-circuit `on_mouse_up` while `composer.collapsed` so stale geometry
from the hidden draft cannot activate a paste token:

```python
if composer.collapsed:
    return
```

- [ ] **Step 9: Verify Task 2**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_workbench_pane_focus.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 10: Commit Task 2**

```bash
git add \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_workbench_pane_focus.py \
  tldw_chatbook/UI/Screens/chat_screen.py
git diff --cached --check
git commit -m "feat(console): wire composer reading mode"
```

---

### Task 3: Harden reading position, lifecycle, run, and setup behavior

**Files:**

- Modify: `Tests/UI/test_console_composer_collapse.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify if a regression exposes a defect:
  - `tldw_chatbook/UI/Screens/chat_screen.py`
  - `tldw_chatbook/Widgets/Console/console_composer_bar.py`

**Interfaces:** No new public API. This task proves and hardens the approved lifecycle contracts.

- [ ] **Step 1: Add transcript selection and anchored-tail tests**

Seed enough `ConsoleChatMessage` rows to overflow `#console-native-transcript`, select one with `transcript.select_message(message_id)`, and assert:

```python
assert transcript.is_anchored
assert transcript.scroll_y == transcript.max_scroll_y
selected = transcript.selected_message_id

console._set_console_composer_collapsed(True)
await pilot.pause()
assert transcript.is_anchored
assert transcript.scroll_y == transcript.max_scroll_y
assert transcript.selected_message_id == selected

console._set_console_composer_collapsed(False)
await pilot.pause()
assert transcript.is_anchored
assert transcript.scroll_y == transcript.max_scroll_y
assert transcript.selected_message_id == selected
```

Add a second test:

```python
transcript.release_anchor()
transcript.scroll_to(y=2, animate=False)
await pilot.pause()
reading_y = transcript.scroll_y

console._set_console_composer_collapsed(True)
await pilot.pause()
assert transcript.is_anchored is False
assert transcript.scroll_y == min(reading_y, transcript.max_scroll_y)

console._set_console_composer_collapsed(False)
await pilot.pause()
assert transcript.is_anchored is False
assert transcript.scroll_y == min(reading_y, transcript.max_scroll_y)
```

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_composer_collapse.py \
  -k "anchored or reading_position or selection" --tb=short
```

Expected: PASS. If Textual applies the layout one refresh later, move the restore into one additional `call_after_refresh` guarded by the same revision; do not use sleeps in product code.

- [ ] **Step 2: Add the rapid-toggle race regression**

Exercise collapse then expansion without a pause:

```python
console._set_console_composer_collapsed(True)
console._set_console_composer_collapsed(False)
await pilot.pause()
await pilot.pause()

assert composer.collapsed is False
assert host.focused is composer
assert transcript.selected_message_id == selected
```

Also test Collapse → priority Escape before the first deferred callback settles. The final state must be expanded/focused; a stale collapse callback must not move focus back to the transcript.

- [ ] **Step 3: Add real active/stale Stop integration tests**

In `Tests/UI/test_console_native_chat_flow.py`, use the existing `WaitingGateway`:

```python
console._set_console_composer_collapsed(True)
composer.load_draft("hello")
# Expand only for the send, start WaitingGateway, collapse again while active.
```

Prefer the exact sequence:

1. Load/send draft through the existing visible Send button.
2. Wait for `gateway.started`.
3. Collapse.
4. Assert `#console-collapsed-stop-generation` is displayed and composer height is one.
5. Press collapsed Stop.
6. Assert partial message becomes `stopped`, collapsed state remains true, and the expanded Stop remains hidden with its presentation.

Then force the stale-stop path by letting the run finish after the button was rendered and invoking `_stop_console_generation_from_visible_action()`. Assert the existing `No active Console run to stop.` warning and no expansion.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_native_chat_flow.py \
  -k "collapsed_stop" --tb=short
```

Expected: PASS.

- [ ] **Step 4: Add tab/workspace/navigation retention tests**

Using existing native tab helpers:

1. Collapse in tab A.
2. Switch to tab B with a different draft and attachment.
3. Assert `composer.collapsed is True`, the left status follows tab B, and focus is `#console-composer-expand` rather than the hidden root.
4. Switch back and assert tab A's draft is intact.

Use `ConsoleNavigationHarness` to capture a `NavigateToScreen` event, call `console.on_screen_resume()`, and assert collapse remains true. Construct a fresh `_build_test_app()` plus `ConsoleHarness` and assert its composer starts expanded.

For the recompose boundary:

```python
console._console_composer_collapsed = True
console._sync_console_session_draft()
await console.recompose()
await pilot.pause()
replacement = console.query_one("#console-native-composer", ConsoleComposerBar)
assert replacement.collapsed is True
assert replacement.draft_text() == expected_session_draft
```

Assert only canonical session draft survives; do not assert transient caret or paste-token presentation across recompose.

- [ ] **Step 5: Add real setup-modal blocking tests**

Use the existing missing-key setup harness in `test_console_native_chat_flow.py`:

1. Assert clicking either toggle programmatically while `_console_setup_modal_blocking()` is true does not change state.
2. Assert the collapsed-only Escape action is disabled while blocked.
3. Start collapsed, make setup blocking, and assert the boolean is retained behind the modal.
4. Resolve setup and call the existing focus restoration path; assert focus lands on `#console-composer-expand`, while `#console-native-composer.can_focus` stays false.

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_composer_collapse.py \
  -k "collapsed or composer_collapse" --tb=short
```

Expected: PASS.

- [ ] **Step 6: Add responsive/status-clipping assertions**

At both `size=(140, 42)` and `size=(100, 32)`, create collapsed-with-draft+attachment and collapsed-generating states. Assert:

```python
assert composer.region.height == 1
assert status.region.right <= stop.region.x
assert stop.region.right <= expand.region.x
assert expand.region.right <= composer.region.right
```

At 100×32, allow the status text to ellipsize, but require Stop and Expand to retain nonzero fixed widths. Assert the transcript gains at least four rows relative to expanded mode.

Because the top-area status strip is absent in this checkout, add a source-level assertion that the composer code contains no query or selector dependency on a status-strip id.

- [ ] **Step 7: Run the hardened focused suite**

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_internals_decomposition.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_workbench_pane_focus.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_transcript_tail_follow.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Stage only files actually changed:

```bash
git add \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_native_chat_flow.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/Widgets/Console/console_composer_bar.py
git diff --cached --check
git commit -m "test(console): harden collapsible composer lifecycle"
```

If either product file did not change, omit it from `git add`.

---

### Task 4: Static checks, broader regressions, live QA, and Backlog closeout

**Files:**

- Create: `Docs/superpowers/qa/console-collapsible-composer-2026-07/README.md`
- Create: screenshots under `Docs/superpowers/qa/console-collapsible-composer-2026-07/`
- Modify: `backlog/tasks/task-398 - Add-collapsible-Console-composer-reading-mode.md`

**Interfaces:** No product interfaces. This is the evidence and closeout gate.

- [ ] **Step 1: Run formatting and static checks**

```bash
.venv/bin/ruff check \
  tldw_chatbook/Widgets/Console/console_composer_bar.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_workbench_pane_focus.py \
  Tests/UI/test_console_native_chat_flow.py
.venv/bin/python -m compileall -q \
  tldw_chatbook/Widgets/Console/console_composer_bar.py \
  tldw_chatbook/UI/Screens/chat_screen.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 2: Run broader Console and Chat regressions**

```bash
.venv/bin/python -m pytest -q \
  Tests/Chat \
  Tests/UI/test_console_*.py \
  Tests/UI/test_workbench_pane_focus.py \
  --tb=short
```

Expected: PASS. If unrelated dirty-worktree failures exist, rerun the exact failing file on the pre-feature commit or document the verified baseline before proceeding; no touched-file failure is acceptable.

- [ ] **Step 3: Prepare isolated live-QA profile**

Use an isolated profile so no personal conversations, filenames, or credentials appear:

```bash
mkdir -p /private/tmp/tldw-qa-composer-20260722/.config/tldw_cli
mkdir -p /private/tmp/tldw-qa-composer-20260722/.local/share
```

Create `/private/tmp/tldw-qa-composer-20260722/.config/tldw_cli/config.toml` with `apply_patch`:

```toml
[general]
default_tab = "chat"

[splash_screen]
enabled = false

[console.onboarding]
first_send_completed = true

[chat_defaults]
provider = "llama_cpp"
model = "composer-qa"

[api_settings.llama_cpp]
api_url = "http://127.0.0.1:8898"
model = "composer-qa"
```

Use a deterministic local OpenAI-compatible endpoint on port 8898 which delays its chat-completion response long enough to capture active generation. Keep that scratch server outside the repo and document it in the QA README as a sanctioned QA stub; product run control still traverses the real `ConsoleProviderGateway`/controller path.

Create `/private/tmp/tldw-qa-composer-20260722/slow_llm.py` with
`apply_patch`:

```python
"""Deterministic delayed OpenAI-compatible endpoint for composer live QA."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        payload = {
            "object": "list",
            "data": [{"id": "composer-qa", "object": "model"}],
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        time.sleep(30)
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "composer-qa",
                "object": "chat.completion.chunk",
                "model": "composer-qa",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "QA response complete."},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        response = {
            "id": "composer-qa",
            "object": "chat.completion",
            "model": "composer-qa",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "QA response complete.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


ThreadingHTTPServer(("127.0.0.1", 8898), Handler).serve_forever()
```

Launch it in terminal B:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  /private/tmp/tldw-qa-composer-20260722/slow_llm.py
```

- [ ] **Step 4: Launch the real app through textual-serve**

In terminal A:

```bash
env \
  HOME=/private/tmp/tldw-qa-composer-20260722 \
  XDG_CONFIG_HOME=/private/tmp/tldw-qa-composer-20260722/.config \
  XDG_DATA_HOME=/private/tmp/tldw-qa-composer-20260722/.local/share \
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve \
  --host 127.0.0.1 --port 9198
```

Expected: server announces `http://127.0.0.1:9198`.

Use bundled Playwright Chromium, abort only external `https://**` requests, navigate with `wait_until="commit"`, and wait for `body.-first-byte`. Do not use the system Chrome channel.

- [ ] **Step 5: Capture and inspect six live screenshots**

Create `Docs/superpowers/qa/console-collapsible-composer-2026-07/` and capture:

1. `wide-expanded.png` — 140×42-equivalent viewport, a retained draft visible, composer height five-to-eight rows.
2. `wide-collapsed-draft.png` — same state collapsed; exact one-row `Composer hidden · Draft retained` plus Expand.
3. `wide-collapsed-generating.png` — active delayed run; `Generating`, Stop, and Expand visible.
4. `compact-expanded.png` — 100×32-equivalent viewport.
5. `compact-collapsed-draft.png` — status ellipsizes before Expand clips.
6. `compact-collapsed-generating.png` — Stop and Expand remain legible.

For each state, record xterm cell geometry in the README, not only browser pixels. Verify:

- collapse adds four-to-seven transcript rows;
- the collapsed composer is exactly one terminal row;
- status text contains no draft or filename;
- fixed controls do not overlap;
- Stop works without expansion;
- Escape expands in one press;
- selection/read position remain stable;
- no personal data or key is visible.

Use `functions.view_image` or the active visual-inspection tool to inspect every PNG before staging it.

- [ ] **Step 6: Write QA evidence and request user approval**

Create `Docs/superpowers/qa/console-collapsible-composer-2026-07/README.md` containing:

- branch and commit tested;
- isolated HOME/config;
- textual-serve and Playwright recipe;
- delayed local stub disclosure;
- viewport/cell sizes;
- per-screenshot behavior verified;
- focused and broader test results;
- secret/privacy inspection result;
- `Approval status: pending`.

Present the actual screenshots to the user. Do not mark TASK-398 Done and do not merge until the user explicitly approves the rendered UI.

- [ ] **Step 7: Close TASK-398 after approval**

After approval:

1. Change all five acceptance criteria to checked.
2. Add concise `## Implementation Notes` covering:
   - stable mounted dual presentation;
   - screen-owned transient state;
   - dynamic priority Escape and state-aware focus;
   - semantic scroll restoration and race guard;
   - hidden-input/confirmation safety;
   - CSS bundle regeneration;
   - tests and approved screenshot paths;
   - ADR required: no; ADR-011 applies.
3. Set the task Done:

```bash
backlog task edit 398 -s Done --plain
```

Verify:

```bash
backlog task 398 --plain
```

Expected: status `Done`, all AC boxes checked, Implementation Plan present, Implementation Notes present, and ADR decision recorded.

- [ ] **Step 8: Commit QA and closeout**

```bash
git add \
  Docs/superpowers/qa/console-collapsible-composer-2026-07 \
  "backlog/tasks/task-398 - Add-collapsible-Console-composer-reading-mode.md"
git diff --cached --check
git commit -m "docs(console): verify collapsible composer"
```

- [ ] **Step 9: Finish the branch**

Invoke `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Include the spec, plan, TASK-398, test commands/results, and approved wide/compact screenshots in the handoff or PR description.

---

## Self-Review Notes

- Every acceptance criterion maps to implementation plus mounted coverage:
  - AC1: Task 1 geometry/presentation/CSS and Task 3 responsive tests.
  - AC2: Task 1 stable widget state plus Task 2 Unfurl/unknown-command safety.
  - AC3: Task 2 input/focus/Escape/F6 plus Task 3 scroll/selection/Stop.
  - AC4: Task 2 screen-owned state plus Task 3 tab/navigation/new-instance/recompose tests.
  - AC5: Tasks 1–3 mounted suites and Task 4 broader verification.
- The plan preserves the existing non-priority Escape path and uses verified Textual dynamic-action semantics for the new priority path.
- Both generic focus methods use the same state-aware resolver, preventing later focus restoration from re-enabling the hidden root.
- `_focus_console_composer_if_needed(force=True)` is included because tab/session paths call it directly.
- `_apply_console_setup_block` is included because its existing `can_focus = not blocking` assignment would otherwise re-enable a collapsed root after setup resolves.
- Draft presence deliberately uses `bool(draft_text())`, so a whitespace-only retained draft is disclosed without exposing content.
- Run state updates the contextual collapsed Stop through the existing action-state sync; it does not add a second controller path.
- Scroll restoration distinguishes anchored tail-follow from manual reading offsets and clamps only after layout.
- Deferred callbacks carry both a monotonic revision and expected boolean, covering rapid Collapse → Expand/Escape.
- Full screen recompose is explicitly treated as a remount boundary: collapsed layout and canonical session draft recover, transient editor internals are not falsely promised.
- No placeholders, persistence additions, new dependencies, selector renames, or unrelated refactors are in scope.
