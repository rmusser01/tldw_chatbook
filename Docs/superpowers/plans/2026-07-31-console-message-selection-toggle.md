# Console Message Selection Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Console users deselect the active transcript message by activating that same message again with either the mouse or keyboard.

**Architecture:** Add one explicit toggle operation to `ConsoleTranscript` and route direct mouse/keyboard message activation through it. Keep `select_message()` as the idempotent absolute-selection API used by navigation and internal action-focus paths, while reusing the existing clear/select refresh and notification behavior.

**Tech Stack:** Python 3.11+, Textual, pytest, pytest-asyncio, Textual `App.run_test()` and Pilot.

**Backlog task:** `TASK-1334`

**Design spec:** `Docs/superpowers/specs/2026-07-31-console-message-selection-toggle-design.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a routine interaction correction inside the established Console transcript selection boundary. It changes no storage, ownership, service contract, security policy, dependency, or long-lived application structure.

## Global Constraints

- Clicking the selected message and pressing Enter on the selected transcript message must both clear selection and hide the contextual action row.
- Enter with no selection must still select the first transcript message.
- Enter on a focused contextual action button must continue to activate the button without clearing selection.
- Up/Down and J/K navigation must remain absolute selection; boundary navigation must not toggle a message off.
- `select_message(message_id: str) -> None` remains idempotent for internal callers.
- Unknown or stale message IDs remain no-ops.
- Do not change CSS, persistence, message models, screen-level state, or message-action behavior.
- Follow strict red-green-refactor: observe each new behavior test fail for the expected missing branch before editing production code.

---

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Widgets/Console/console_transcript.py` | Own the selection toggle and route message-row clicks and transcript Enter activation through it. |
| `Tests/UI/test_console_native_transcript.py` | Exercise real mounted Textual mouse and keyboard behavior with `TranscriptHarness`. |
| `backlog/tasks/task-1334 - Toggle-selected-Console-message-off-on-repeated-activation.md` | Track acceptance criteria, ADR determination, verification, and implementation notes. |

No new production modules or dependencies are required.

---

### Task 1: Toggle Selection from a Repeated Message-Row Click

**Files:**
- Modify: `Tests/UI/test_console_native_transcript.py:458-489`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:237-243,549-556`

**Interfaces:**
- Consumes: `ConsoleTranscript.selected_message_id: str | None`, `ConsoleTranscript.select_message(message_id: str) -> None`, and `ConsoleTranscript.action_clear_selection() -> None`.
- Produces: `ConsoleTranscript.toggle_message_selection(message_id: str) -> None`, used by the message-row click handler and Task 2's keyboard handler.

- [ ] **Step 1: Add the failing mounted mouse regression**

Add this test after `test_console_transcript_click_selects_message_and_shows_actions`. The production mutation it catches is `ConsoleTranscriptMessage.on_click()` using absolute selection for an already selected row.

```python
@pytest.mark.asyncio
async def test_console_transcript_click_selected_message_clears_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one(
            "#console-native-transcript", ConsoleTranscript
        )

        await pilot.click("#console-message-m2")
        await pilot.pause()
        assert transcript.selected_message_id == "m2"
        assert "Save as..." in _visible_text(app)

        await pilot.click("#console-message-m2")
        await pilot.pause()

        assert transcript.selected_message_id is None
        assert "Save as..." not in _visible_text(app)
```

- [ ] **Step 2: Run the mouse regression and verify RED**

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_click_selected_message_clears_selection -v
```

Expected: FAIL on `assert transcript.selected_message_id is None` because the second click leaves it equal to `"m2"`.

- [ ] **Step 3: Add the minimal toggle operation and route row clicks through it**

Add the method directly after `select_message()`:

```python
def toggle_message_selection(self, message_id: str) -> None:
    """Toggle one message's contextual selection state."""
    if self._message_by_id(message_id) is None:
        return
    if self.selected_message_id == message_id:
        self.action_clear_selection()
        return
    self.select_message(message_id)
```

Change the final line of `ConsoleTranscriptMessage.on_click()`:

```python
if isinstance(transcript, ConsoleTranscript):
    transcript.toggle_message_selection(self.message_id)
```

Do not alter `select_message()`; navigation and internal callers require absolute selection.

- [ ] **Step 4: Run the mouse regression and verify GREEN**

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_click_selected_message_clears_selection -v
```

Expected: PASS.

- [ ] **Step 5: Run neighboring mouse-selection regressions**

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py -k "click_selects_message or click_selected_message or click_background or click_action_button or click_rule_separator" -v
```

Expected: all selected tests PASS; action buttons, rule separators, and negative-space clicks retain their existing behavior.

- [ ] **Step 6: Commit the mouse toggle**

```bash
git add Tests/UI/test_console_native_transcript.py tldw_chatbook/Widgets/Console/console_transcript.py
git commit -m "feat(console): toggle selected message on click"
```

---

### Task 2: Toggle Selection from Keyboard Enter

**Files:**
- Modify: `Tests/UI/test_console_native_transcript.py:439-455`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:615-617`

**Interfaces:**
- Consumes: `ConsoleTranscript.toggle_message_selection(message_id: str) -> None` from Task 1 and the existing `ConsoleTranscriptActionButton.on_key()` event stop.
- Produces: Enter semantics where no selection selects the first message and an existing transcript selection toggles off.

- [ ] **Step 1: Replace the misleading keyboard test with explicit behavior tests**

Replace `test_console_transcript_keyboard_selects_messages_and_enter_shows_actions` with these tests. The first characterizes the existing no-selection contract. The second catches the missing selected-message Enter branch.

```python
@pytest.mark.asyncio
async def test_console_transcript_enter_selects_first_message_when_none_selected():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        transcript.focus()

        await pilot.press("enter")
        await pilot.pause()

        assert transcript.selected_message_id == "m1"
        assert "Save as..." in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_enter_clears_keyboard_selected_message():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        transcript.focus()

        await pilot.press("down")
        await pilot.pause()
        assert transcript.selected_message_id == "m1"
        assert "Save as..." in _visible_text(app)

        await pilot.press("enter")
        await pilot.pause()

        assert transcript.selected_message_id is None
        assert "Save as..." not in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_boundary_navigation_keeps_last_message_selected():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        transcript.focus()

        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()

        assert transcript.selected_message_id == "m2"
        assert "Save as..." in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_enter_on_action_button_preserves_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        await pilot.click("#console-message-m2")
        transcript.focus_action("m2", "copy")
        await pilot.pause()

        button = app.query_one("#console-message-action-copy-m2", Button)
        assert button.has_focus

        await pilot.press("enter")
        await pilot.pause()

        assert transcript.selected_message_id == "m2"
        assert "Save as..." in _visible_text(app)
```

- [ ] **Step 2: Verify the new keyboard expectations before production changes**

Run the failing toggle test:

```bash
pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_enter_clears_keyboard_selected_message -v
```

Expected: FAIL because Enter leaves `selected_message_id == "m1"`.

Run the preserved no-selection test:

```bash
pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_enter_selects_first_message_when_none_selected -v
```

Expected: PASS before production changes, confirming the existing behavior that must be preserved.

Run the navigation and action-button characterizations:

```bash
pytest Tests/UI/test_console_native_transcript.py -k "boundary_navigation_keeps_last_message_selected or enter_on_action_button_preserves_selection" -v
```

Expected: both tests PASS before production changes, proving the absolute-selection and focused-button event boundaries that the implementation must retain.

- [ ] **Step 3: Route transcript Enter through the toggle operation**

Replace `action_confirm_selection()` with:

```python
def action_confirm_selection(self) -> None:
    if self.selected_message_id is not None:
        self.toggle_message_selection(self.selected_message_id)
        return
    if self._messages:
        self.select_message(self._messages[0].id)
```

Do not change `ConsoleTranscriptActionButton.on_key()`; it already presses the focused button and stops the Enter event before the transcript can treat it as a selection toggle.

- [ ] **Step 4: Run the keyboard and preserved-boundary tests and verify GREEN**

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py -k "enter_selects_first_message_when_none_selected or enter_clears_keyboard_selected_message or boundary_navigation_keeps_last_message_selected or enter_on_action_button_preserves_selection" -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Verify navigation and action-row input boundaries**

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py -k "keyboard or boundary_navigation or action_button_preserves_selection or enter_on_action_button or escape_collapses_selected_action_row" -v
```

Expected: all selected tests PASS. Up/Down and J/K still select absolutely, Escape still clears, and contextual action-button activation preserves selection.

- [ ] **Step 6: Commit the keyboard toggle**

```bash
git add Tests/UI/test_console_native_transcript.py tldw_chatbook/Widgets/Console/console_transcript.py
git commit -m "feat(console): toggle selected message with enter"
```

---

### Task 3: Verify and Close TASK-1334

**Files:**
- Modify: `backlog/tasks/task-1334 - Toggle-selected-Console-message-off-on-repeated-activation.md`

**Interfaces:**
- Consumes: the completed mouse and keyboard behavior from Tasks 1-2.
- Produces: verified acceptance criteria, implementation notes, and a Done Backlog task.

- [ ] **Step 1: Run the complete focused transcript module**

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py -q --tb=short
```

Expected: PASS with no errors or warnings attributable to the change.

- [ ] **Step 2: Run the broader selected-message Console regressions**

Run:

```bash
pytest Tests/UI/test_console_native_chat_flow.py -k "selected_message or message_action" -q --tb=short
```

Expected: all selected tests PASS.

- [ ] **Step 3: Run static diff validation and self-review**

Run:

```bash
git diff --check
```

Expected: no output and exit status 0.

Review the final diff and confirm it contains only the explicit toggle method, the two activation routes, focused tests, and task documentation. Mentally mutate each route back to `select_message()` and confirm the corresponding mouse or keyboard test would fail.

- [ ] **Step 4: Complete the Backlog task**

Check all five acceptance criteria, add concise implementation notes naming the toggle API, mouse/keyboard routing, tests, and verification results, preserve the recorded ADR decision (`ADR required: no`), then set the task to Done:

```bash
backlog task edit 1334 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --notes "Added an explicit Console transcript selection toggle while preserving idempotent navigation selection. Repeated message-row clicks and Enter on the selected transcript message now clear selection through the existing refresh/notification path; Enter with no selection and focused contextual action buttons retain their prior behavior. Added mounted Textual regressions for mouse and keyboard paths and verified the focused transcript and selected-message Console suites. ADR required: no; this change stays within the existing UI interaction boundary. Modified tldw_chatbook/Widgets/Console/console_transcript.py and Tests/UI/test_console_native_transcript.py." -s Done
```

- [ ] **Step 5: Commit task completion metadata**

```bash
git add "backlog/tasks/task-1334 - Toggle-selected-Console-message-off-on-repeated-activation.md"
git commit -m "docs(backlog): complete TASK-1334"
```
