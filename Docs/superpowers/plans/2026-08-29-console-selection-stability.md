# Console Selection Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Console selection-menu replacement pruning-safe and preserve the initially pressed message across layout-shifting menu dismissal.

**Architecture:** Keep the two existing lifecycle mechanisms separate: settled dismissal continues through the non-pruning registry view, while same-ID remount awaits an unfiltered public screen query. For plain row gestures, classify and latch the row before selection-UI cleanup, commit its immutable message ID on empty MouseUp, and reuse the manager's existing one-shot suppression flag to absorb an optional later Click.

**Tech Stack:** Python 3.11+, Textual 8.x mouse/message lifecycle, pytest, pytest-asyncio

**Backlog:** `TASK-24529` followed by dependent `TASK-24530`

**ADR required:** no

**ADR path:** N/A

**Reason:** Both changes correct event ordering inside the existing screen-owned selection lifecycle and add no persistence, dependency, or cross-module contract.

---

## File Map

- Modify `tldw_chatbook/Widgets/Console/console_transcript.py`: await every attached menu at the remount boundary; preserve press identity through empty MouseUp; release capture on Escape.
- Modify `Tests/UI/test_console_selection_menu.py`: add a deterministic already-pruning/no-yield remount regression and retain settled menu lifecycle controls.
- Modify `Tests/UI/test_console_selection_transcript.py`: add raw App MouseDown/MouseUp, exact-once Click, layout-sensitive targeting, branch, and cancellation regressions.
- Modify `backlog/tasks/task-24529 - Await-pruning-Console-selection-menu-remounts.md`: record the final implementation evidence and close the prerequisite task only after its focused tests pass.
- Modify `backlog/tasks/task-24530 - Preserve-the-initial-Console-row-click-target.md`: record the final implementation evidence and close the click-target task only after the complete ordered Console slice passes.

No new production file, helper, manager, timeout, or dependency is planned.

## Pre-implementation formatter baseline

Before changing production or tests, run:

```bash
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  Tests/UI/test_console_selection_menu.py \
  Tests/UI/test_console_selection_transcript.py
```

Expected inherited baseline: exit 1 naming exactly all three files as requiring
whole-file reformat. Current Ruff output is approximately 432, 171, and 211
diff lines respectively; bulk-formatting these legacy files would create a
large unrelated diff. Preserve this exact three-file failure set, keep every
task-owned hunk formatter-consistent, and record the inherited deviation rather
than claiming a green whole-file formatter gate.

### Task 1: Pin and fix the already-pruning menu remount race (`TASK-24529`)

**Files:**
- Modify: `Tests/UI/test_console_selection_menu.py:180-230`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:5273-5320`

- [ ] **Step 1: Replace the settled-only claim with a deterministic no-yield regression**

Add a sibling test that holds the old object, schedules removal without awaiting it, proves the filtered and unfiltered views disagree, and calls the handler directly before any `pilot.pause()` can settle pruning:

```python
@pytest.mark.asyncio
async def test_pruning_menu_is_awaited_before_same_id_remount():
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        transcript = app.query_one(ConsoleTranscript)
        old_menu = app.query_one(ConsoleSelectionMenu)
        old_menu.remove()  # schedule pruning; deliberately do not await

        assert old_menu.is_attached
        assert old_menu._pruning is True
        assert old_menu not in transcript._attached_selection_menus()
        assert old_menu in transcript.screen.query(ConsoleSelectionMenu)

        row = app.query_one("#console-message-m1", ConsoleTranscriptMessage)
        selection = TextSelection(row.id, 0, 5)
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        transcript.selection_manager.finish_drag()
        await transcript._text_selected(
            ConsoleTranscript.TranscriptTextSelected(
                selection=selection,
                screen_x=4,
                screen_y=6,
            )
        )

        replacement = app.query_one(ConsoleSelectionMenu)
        assert old_menu.is_attached is False
        assert replacement is not old_menu
        assert replacement._pruning is False
        assert len(app.query(ConsoleSelectionMenu)) == 1
        assert app.is_running

        await pilot.pause()
        assert app.query_one(ConsoleSelectionMenu) is replacement
        assert app.is_running
```

Update the existing `test_consecutive_selections_remount_exactly_one_menu` docstring to describe it as settled interaction coverage, not the no-yield race proof.

- [ ] **Step 2: Run the race test and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_selection_menu.py::test_pruning_menu_is_awaited_before_same_id_remount
```

Expected: FAIL before the fix with `textual.dom.DuplicateIds` when `_text_selected` mounts a second `#console-selection-menu` while the old object is still attached and `_pruning`.

- [ ] **Step 3: Await the unfiltered public query only at the remount boundary**

In `_text_selected`, replace the filtered per-menu loop:

```python
for menu in self._attached_selection_menus():
    await menu.remove()
```

with the existing Textual query operation:

```python
await self.screen.query(ConsoleSelectionMenu).remove()
```

Do not change `_attached_selection_menus()` or `_remove_selection_menu()`; their non-pruning fire-and-forget semantics are intentional.

- [ ] **Step 4: Run the new race proof and settled controls**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_selection_menu.py::test_pruning_menu_is_awaited_before_same_id_remount \
  Tests/UI/test_console_selection_menu.py::test_consecutive_selections_remount_exactly_one_menu \
  Tests/UI/test_console_selection_menu.py::test_escape_dismisses_menu_in_transcript_context \
  Tests/UI/test_console_selection_menu.py::test_add_to_chat_quotes_selection_and_cleans_up
```

Expected: 4 passed; the old object is detached before the replacement mounts and ordinary dismissal behavior remains intact.

- [ ] **Step 5: Commit the prerequisite fix**

```bash
git add -- Tests/UI/test_console_selection_menu.py tldw_chatbook/Widgets/Console/console_transcript.py
git diff --cached --check
git commit -m "fix: await pruning Console selection menus"
```

### Task 2: Pin raw missing-Click and optional-Click exact-once behavior (`TASK-24530`)

**Files:**
- Modify: `Tests/UI/test_console_selection_transcript.py:535-790`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:5152-5270`

- [ ] **Step 1: Strengthen the existing pilot regression as the optional-Click exact-once proof**

Keep `test_menu_open_row_body_click_dismisses_menu_and_toggles`, but extend its final assertions:

```python
assert not app.query(ConsoleSelectionMenu)
assert transcript.selected_message_id == "m2"
assert transcript.selection_manager.state.active is False
assert transcript.selection_manager.just_finished is False
assert transcript._selection_origin_row is None
```

The current `pilot.click` always injects `[MouseDown, MouseUp, Click]`; selecting `m2` after that sequence proves MouseUp committed once and the injected Click did not toggle it back off.

- [ ] **Step 2: Add the real App MouseDown/MouseUp regression with no Click**

Add a module-level helper beside `_mouse_event` so raw terminal-shaped tests do not repeat constructor details:

```python
def _raw_app_mouse(event_cls, screen_x: int, screen_y: int, *, button: int = 1):
    return event_cls(
        widget=None,
        x=screen_x,
        y=screen_y,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=screen_x,
        screen_y=screen_y,
    )
```

Use it to add the no-Click and no-replay proof:

```python
@pytest.mark.asyncio
async def test_menu_cleanup_raw_mouseup_commits_initial_press_without_click():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        selected_row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, selected_row, start_x=3, end_x=11)
        assert app.query_one(ConsoleSelectionMenu)

        target = app.query_one("#console-message-m2", ConsoleTranscriptMessage)
        target_body = _body_static(target)
        x, y = target_body.region.x + 1, target_body.region.y + 1
        app.post_message(_raw_app_mouse(MouseDown, x, y))
        await pilot.pause()
        assert transcript._selection_origin_row is target

        app.post_message(_raw_app_mouse(MouseUp, x, y))
        await pilot.pause()
        assert transcript.selected_message_id == "m2"
        assert transcript._selection_origin_row is None

        first = app.query_one("#console-message-m1", ConsoleTranscriptMessage)
        x2, y2 = first.region.x + 1, first.region.bottom - 1
        app.post_message(_raw_app_mouse(MouseDown, x2, y2))
        await pilot.pause()
        app.post_message(_raw_app_mouse(MouseUp, x2, y2))
        await pilot.pause()
        assert transcript.selected_message_id == "m1"
        assert transcript._selection_origin_row is None
```

Do not post a `Click` in this test. Its first MouseDown must also assert the old menu/highlight disappeared so the geometry-changing cleanup genuinely ran.

- [ ] **Step 3: Run both outcome tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_selection_transcript.py::test_menu_open_row_body_click_dismisses_menu_and_toggles \
  Tests/UI/test_console_selection_transcript.py::test_menu_cleanup_raw_mouseup_commits_initial_press_without_click
```

Expected before the fix: the existing pilot case leaves `m1` selected, and the raw App case leaves the message selection unchanged because no Click is synthesized.

- [ ] **Step 4: Classify before cleanup and commit empty gestures on MouseUp**

Reorder `on_mouse_down` so it resolves the row before selection UI can move it, latches that row before cleanup, clears the previous highlight before `begin_drag()` replaces the manager's finished selection, then starts the normal drag:

```python
press_control = self._selection_press_widget(event)
press_node: Widget | None = press_control
while press_node is not None and not isinstance(
    press_node, ConsoleSelectionMenu
):
    press_node = press_node.parent

row = self._selection_row_for(press_control) if event.button == 1 else None
if press_node is None:
    if row is not None:
        self._selection_origin_row = row
    self._remove_selection_menu()
    if row is not None:
        self._clear_other_selection_highlights(row)

if row is None:
    self.selection_manager.consume_just_finished()
    return

offset = self._selection_offset_for(row, event.screen_x, event.screen_y)
self.selection_manager.begin_drag(row.id, offset)
self._selection_origin_row = row
self.capture_mouse(True)
```

Remove the old post-`begin_drag()` `_clear_other_selection_highlights(row)` call so cleanup does not run twice.

In `on_mouse_up`, preserve the origin until after `finish_drag()` and toggle its immutable model ID for an empty gesture:

```python
origin_row = self._selection_origin_row
selection = self.selection_manager.finish_drag()
self._selection_origin_row = None
if selection is None:
    if origin_row is not None:
        self.toggle_message_selection(origin_row.message_id)
    return
```

Do not consume `just_finished` on this branch. Existing row/transcript Click guards must consume it if an optional Click follows. The validated toggle already makes a removed message a safe no-op.

- [ ] **Step 5: Run the raw and pilot outcome tests and verify GREEN**

Run the same two-test command from Step 3.

Expected: 2 passed; raw MouseUp selects the original target without Click, and `pilot.click` leaves that target selected exactly once.

### Task 3: Pin cancellation and reordered classification branches

**Files:**
- Modify: `Tests/UI/test_console_selection_transcript.py:535-790`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:5152-5270, 5760-5780`

- [ ] **Step 1: Add an Escape lifecycle regression through the public pilot API**

Open a menu, press another selectable row with `pilot.mouse_down`, and assert the armed state before cancelling:

```python
await pilot.mouse_down("#console-message-m2", offset=(1, 1))
await pilot.pause()
assert transcript.selection_manager.state.active is True
assert app.mouse_captured is transcript
assert transcript._selection_origin_row.message_id == "m2"
assert not app.query(ConsoleSelectionMenu)
assert selected_row.get_selection_text() == ""

await pilot.press("escape")
await pilot.pause()
assert transcript.selection_manager.is_idle
assert transcript._selection_origin_row is None
assert app.mouse_captured is None
assert not app.query(ConsoleSelectionMenu)
assert selected_row.get_selection_text() == ""
```

- [ ] **Step 2: Add direct right-button and menu-descendant branch controls**

For the right-button case, open a menu and call `pilot.mouse_down` on a selectable row with `button=3`; assert immediate menu dismissal, no origin, and an idle manager.

For the menu guard, route a synthetic event directly through the transcript so the branch cannot be intercepted by the screen-mounted menu:

```python
menu = app.query_one(ConsoleSelectionMenu)
button = menu.query_one("#console-selection-add-to-chat")
event = _mouse_event(
    MouseDown,
    button,
    screen_x=button.region.x + 1,
    screen_y=button.region.y,
)
transcript.on_mouse_down(event)
assert menu.is_attached
assert transcript._selection_origin_row is None
assert transcript.selection_manager.state.active is False
```

- [ ] **Step 3: Add same-row and Markdown layout-sensitive controls**

Reuse the existing plain-row and assistant Markdown fixtures:

- select text and open the menu on `m1`, then `pilot.click` the same row body and assert `m1` is selected with no menu or highlight;
- select assistant Markdown text, then click a different selectable plain row and assert the initially pressed row is selected with no menu or stale Markdown selection strip.

These tests must assert both the selected message ID and cleared selection text; a message-only assertion is insufficient.

- [ ] **Step 4: Verify RED for incomplete Escape cleanup**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_selection_transcript.py -k "escape_clears_armed_press or right_button or menu_descendant or same_row or markdown_layout"
```

Expected before the final cleanup: the Escape test fails because the transcript still owns mouse capture. Branch tests may already pass and serve as non-regression controls.

- [ ] **Step 5: Complete Escape cleanup minimally**

In the ordinary Escape branch of `ConsoleTranscript.on_key`, release capture before clearing state:

```python
elif event.key == "escape":
    self.release_mouse()
    self.action_clear_selection()
    self._remove_selection_menu()
    self.selection_manager.cancel()
    self._selection_origin_row = None
    event.stop()
```

Do not change the keyboard-selection-mode Escape branch, which intentionally has different first-Escape behavior.

- [ ] **Step 6: Run the classification/cancellation tests and the deterministic baseline regression**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_selection_transcript.py::test_menu_open_row_body_click_dismisses_menu_and_toggles
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_selection_transcript.py \
  -k "escape_clears_armed_press or right_button or menu_descendant or same_row or markdown_layout or raw_mouseup"
```

Expected: all selected cases pass, including the original deterministic failure.

- [ ] **Step 7: Commit the click-target correction**

```bash
git add -- Tests/UI/test_console_selection_transcript.py tldw_chatbook/Widgets/Console/console_transcript.py
git diff --cached --check
git commit -m "fix: preserve Console row press targets"
```

### Task 4: Verify the complete ordered Console slice and close both tasks

**Files:**
- Test: `Tests/UI/test_console_selection_menu.py`
- Test: `Tests/UI/test_console_selection_transcript.py`
- Test: `Tests/UI/test_console_selection_dismissal_perf.py`
- Test: `Tests/UI/test_console_selection_end_to_end.py`
- Test: `Tests/UI/test_console_keyboard_selection.py`
- Test: `Tests/UI/test_console_selection_rows.py`
- Modify: `backlog/tasks/task-24529 - Await-pruning-Console-selection-menu-remounts.md`
- Modify: `backlog/tasks/task-24530 - Preserve-the-initial-Console-row-click-target.md`

- [ ] **Step 1: Run the focused Console selection suites**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_selection_menu.py \
  Tests/UI/test_console_selection_transcript.py \
  Tests/UI/test_console_selection_dismissal_perf.py \
  Tests/UI/test_console_selection_end_to_end.py \
  Tests/UI/test_console_keyboard_selection.py \
  Tests/UI/test_console_selection_rows.py
```

Expected: all selected tests pass. Do not run the repository-wide suite without explicit user opt-in.

- [ ] **Step 2: Run scoped static checks**

Run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  Tests/UI/test_console_selection_menu.py \
  Tests/UI/test_console_selection_transcript.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  Tests/UI/test_console_selection_menu.py \
  Tests/UI/test_console_selection_transcript.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  Tests/UI/test_console_selection_menu.py \
  Tests/UI/test_console_selection_transcript.py
git diff --check
```

Expected: Ruff check, compilation, and `git diff --check` exit 0. Ruff format
retains exactly the same inherited three-file failure set recorded before
implementation, with no additional file or task-owned formatting drift. Do not
bulk-reformat these legacy modules or claim a green whole-file formatter check.

- [ ] **Step 3: Self-review the exact diff**

Run:

```bash
git diff -- \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  Tests/UI/test_console_selection_menu.py \
  Tests/UI/test_console_selection_transcript.py
```

Confirm the remount path uses the public awaited query, ordinary dismissal still filters `_pruning`, MouseUp owns plain-click activation, no new state exists, and exception handling was not broadened.

- [ ] **Step 4: Complete task records only after evidence is green**

For each Backlog task:

1. check every acceptance criterion;
2. add concise Implementation Notes naming the approach, modified files, targeted commands, results, the exact inherited three-file formatter baseline, and `ADR required: no`;
3. note that no generalizable lesson was added unless implementation actually uncovered one; and
4. set status to Done with `backlog task edit <id> -s Done`, then verify the CLI-reported file path.

- [ ] **Step 5: Commit task closeout**

```bash
git add -- \
  "backlog/tasks/task-24529 - Await-pruning-Console-selection-menu-remounts.md" \
  "backlog/tasks/task-24530 - Preserve-the-initial-Console-row-click-target.md"
git diff --cached --check
git commit -m "docs: close Console selection stability tasks"
```
