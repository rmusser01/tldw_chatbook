# Console Screen UX HCI Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Console screen from a cleaner terminal layout into a more task-obvious, scan-friendly, interaction-focused control surface.

**Architecture:** Keep the current Textual-native Console structure and improve hierarchy in place. Use existing screen orchestration in `ChatScreen`, existing Console widgets for transcript, composer, rail handles, and display-state builders, with CSS token changes kept mirrored in both Console stylesheet files. Every visual change gets a failing widget/layout test first, then a focused implementation, then screenshot QA.

**Tech Stack:** Python 3.11+, Textual, Rich `Text`, pytest, Textual pilot tests, existing Playwright/Textual Serve screenshot harnesses.

---

## Issue Coverage Matrix

| Review Item | Plan Task |
|---|---|
| 1. Center canvas feels inert when empty | Task 1 |
| 2. Composer reads secondary despite being primary | Task 2 |
| 3. Side rail open controls are discoverable but spatially awkward | Task 3 |
| 4. Frame density is still heavy | Task 4 |
| 5. Top status bars are too flat | Task 5 |
| 6. Text controls should use consistent symbols where appropriate | Task 6 |
| 7. Empty left rail is useful but repetitive | Task 7 |
| 8. Tab strip is cramped against transcript title | Task 8 |
| 9. Button priority is not quite right | Task 9 |

## File Structure

- Modify `tldw_chatbook/UI/Screens/chat_screen.py`
  - Owns Console layout composition, control state sync, guidance dismissal, provider blocker, rail composition, and composer action state wiring.
- Modify `tldw_chatbook/Widgets/Console/console_session_surface.py`
  - Owns transcript title, session tab strip, and inline first-run guidance rendering.
- Modify `tldw_chatbook/Widgets/Console/console_transcript.py`
  - Owns transcript empty state copy and message surface rendering.
- Modify `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  - Owns composer focus/draft state classes, visible draft styling, and action button state styling.
- Modify `tldw_chatbook/Widgets/Console/console_rail_handle.py`
  - Owns collapsed rail handle labels, badges, and accessible/tooltipped open affordances.
- Modify `tldw_chatbook/Widgets/Console/console_staged_context.py`
  - Owns staged context empty state, attach affordance, and tray framing behavior.
- Modify `tldw_chatbook/Chat/console_display_state.py`
  - Owns user-facing summary copy for staged context, inspector, source readiness, and empty states.
- Modify `tldw_chatbook/Chat/console_rail_state.py`
  - Owns rail labels, badges, first-start defaults, and collapsed handle labels.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Component-level Console styles.
- Modify `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Mirrored runtime stylesheet rules for the same selectors.
- Modify `Tests/UI/test_console_internals_decomposition.py`
  - Main widget geometry, copy, action state, and composer/transcript tests.
- Modify `Tests/UI/test_console_persistent_rails.py`
  - Rail default, persisted state, collapsed handle, and responsive rail behavior tests.
- Modify screenshot QA artifacts under `Docs/superpowers/qa/console-persistent-rails/` after implementation.

## Task 1: Make the Empty Transcript Canvas Feel Alive, Not Blank

**Addresses:** Item 1.

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`

- [ ] **Step 1: Write a failing empty transcript test**

Add a test named `test_console_empty_transcript_uses_compact_ready_state`.

Expected behavior:
- Empty transcript shows exactly one compact row, `No messages yet. Composer ready.`
- The old first-run long guidance does not appear in the transcript.
- The empty row sits directly below the tab strip.

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_empty_transcript_uses_compact_ready_state --tb=short
```

Expected: FAIL because the current empty text is still only `No messages yet.`.

- [ ] **Step 2: Implement the compact empty state**

In `console_transcript.py`, change the empty transcript render path to emit one row of muted system text:

```python
EMPTY_TRANSCRIPT_COPY = "No messages yet. Composer ready."
```

Keep this copy in the transcript only. Do not restore large onboarding blocks in `ChatScreen`.

- [ ] **Step 3: Verify Task 1**

Run the test above again.

Expected: PASS.

## Task 2: Make Composer Focus and Draft State Unmistakable

**Addresses:** Item 2.

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_internals_decomposition.py`

- [ ] **Step 1: Write failing composer state tests**

Add tests:
- `test_console_composer_marks_focus_state`
- `test_console_composer_marks_has_draft_state`

Expected behavior:
- When the composer is focused, `#console-native-composer` has class `console-composer-focused`.
- When draft text is non-empty, it has class `console-composer-has-draft`.
- After `clear_draft()`, `console-composer-has-draft` is removed.

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_composer_marks_focus_state Tests/UI/test_console_internals_decomposition.py::test_console_composer_marks_has_draft_state --tb=short
```

Expected: FAIL because these state classes do not exist yet.

- [ ] **Step 2: Add composer state synchronization**

In `ConsoleComposerBar`, add a private method:

```python
def _sync_interaction_classes(self) -> None:
    self.set_class(self.has_focus_within, "console-composer-focused")
    self.set_class(bool(self.draft_text().strip()), "console-composer-has-draft")
```

Call it from:
- `on_mount`
- `on_focus`
- `on_blur`
- `load_draft`
- `clear_draft`
- `insert_text`
- `insert_pasted_text`
- `delete_left`

- [ ] **Step 3: Add focused and draft CSS states**

In both CSS files, add rules for:
- `#console-native-composer.console-composer-focused`
- `#console-native-composer.console-composer-has-draft`
- `#console-command-visible-text`

Design intent:
- Focused composer gets the strongest active border on the screen.
- Has-draft state makes the Send button eligible for primary emphasis in Task 9.
- Placeholder remains subdued.

- [ ] **Step 4: Verify Task 2**

Run the two new tests and inspect the 1920 screenshot.

## Task 3: Replace Awkward Rail Open Controls With Proper Handles

**Addresses:** Item 3.

**Files:**
- Modify: `tldw_chatbook/Chat/console_rail_state.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_persistent_rails.py`

- [ ] **Step 1: Write failing collapsed rail handle tests**

Add or update tests so collapsed rails assert:
- Left collapsed handle copy is `Context >`.
- Right collapsed handle copy is `< Inspector`.
- The open buttons have tooltips `Open Context rail` and `Open Inspector rail`.
- Handles are aligned with the workbench frame and do not look like free-floating buttons.

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected: FAIL on current handle copy.

- [ ] **Step 2: Update rail state labels**

In `console_rail_state.py`, change collapsed labels to side-specific handle labels.

- [ ] **Step 3: Update `ConsoleRailHandle` rendering**

Make `ConsoleRailHandle` render a compact handle with:
- one clear label
- optional badge
- full-height hit area
- consistent side-specific classes

- [ ] **Step 4: Verify Task 3**

Run `Tests/UI/test_console_persistent_rails.py`.

Expected: PASS.

## Task 4: Reduce One Layer of Excess Framing

**Addresses:** Item 4.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/Widgets/Console/console_staged_context.py`
- Modify: both Console CSS files
- Test: `Tests/UI/test_console_internals_decomposition.py`

- [ ] **Step 1: Write failing frame hierarchy tests**

Add `test_console_empty_regions_do_not_stack_nested_terminal_frames`.

Expected behavior:
- The workbench keeps a visible outer frame.
- The transcript keeps left, right, and bottom frame boundaries, with no top border.
- Empty staged context does not add a second heavy inner border inside the left rail.
- Composer remains framed because it is the active input surface.

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_empty_regions_do_not_stack_nested_terminal_frames --tb=short
```

Expected: FAIL because current empty left tray still has an inner solid frame.

- [ ] **Step 2: Add frame variants**

In `ChatScreen._frame_console_region`, support a `variant` argument:

```python
def _frame_console_region(widget: Any, *, top: bool = True, variant: str = "solid") -> Any:
```

Use `variant="quiet"` for empty secondary regions and CSS classes for non-critical borders.

- [ ] **Step 3: Apply quiet frame to empty secondary regions**

Apply quiet framing to:
- empty staged context tray
- empty workspace context tray if it has no actionable content
- transcript interior only where needed

- [ ] **Step 4: Verify Task 4**

Run the new frame hierarchy test plus screenshot capture.

## Task 5: Group the Top Status Bar Into Scan-Friendly Segments

**Addresses:** Item 5.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Chat/console_display_state.py`
- Modify: both Console CSS files
- Test: `Tests/UI/test_console_internals_decomposition.py`

- [ ] **Step 1: Write failing status grouping test**

Add `test_console_mode_bar_groups_location_mode_and_readiness`.

Expected rendered text:

```text
Mode: Chat / RAG / Run Follow | Assistant: General | Readiness: Sources 0, Tools 0, Approvals 0
```

The top title row remains:

```text
Console | Live agent control, chat, RAG, tools, approvals | Local
```

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_mode_bar_groups_location_mode_and_readiness --tb=short
```

Expected: FAIL because current mode bar uses flatter pipe-separated copy.

- [ ] **Step 2: Update `_console_mode_summary`**

Refactor `_console_mode_summary` to produce grouped, stable copy. Keep it one row and avoid adding widgets unless Rich span styling is needed.

- [ ] **Step 3: Add optional muted/active span styling**

If using Rich `Text`, style labels muted and values bright. Do not add new height.

- [ ] **Step 4: Verify Task 5**

Run the new test and screenshot QA.

## Task 6: Standardize Symbolic Controls

**Addresses:** Item 6.

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_console_persistent_rails.py`

- [ ] **Step 1: Write failing symbol vocabulary tests**

Expected behavior:
- New tab button label is `+`.
- New tab button tooltip is `New Console tab`.
- Close tab button remains `x` with tooltip `Close Console tab`.
- Rail collapse buttons use `<` and `>` only, with tooltips carrying the descriptive text.
- Visible rail handles use `Context >` and `< Inspector`.

- [ ] **Step 2: Update tab strip buttons**

In `console_session_surface.py`:
- Change `Button("New tab", id="console-new-chat-tab")` to `Button("+", ...)`.
- Set fixed width, tooltip, and compact height.
- Add tooltip to session close buttons.

- [ ] **Step 3: Update rail collapse buttons**

In `chat_screen.py`:
- Change `Hide <` to `<`.
- Change `Hide >` to `>`.
- Keep tooltips.
- Maintain at least 3 columns for hit target in terminal UI.

- [ ] **Step 4: Verify Task 6**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_persistent_rails.py --tb=short
```

Expected: PASS.

## Task 7: Make the Empty Context Rail Actionable

**Addresses:** Item 7.

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_staged_context.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Chat/console_display_state.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`

- [ ] **Step 1: Write failing actionable empty-state test**

Add `test_console_empty_staged_context_exposes_attach_action`.

Expected behavior:
- Empty context rail shows `No staged work.`
- It exposes a compact `Attach` action with id `console-staged-context-attach`.
- Pressing the action routes to the same handler path as the bottom `Attach` button or posts the same event.

- [ ] **Step 2: Implement tray action**

Add a compact button inside the empty staged context tray. Keep copy short:

```text
No staged work.
Attach
```

Do not duplicate `Attach sources.` as static copy when the button is present.

- [ ] **Step 3: Wire the button**

In `ChatScreen`, add an event handler for `#console-staged-context-attach` that reuses `handle_console_attach_context` behavior through a shared helper, not duplicated code.

- [ ] **Step 4: Verify Task 7**

Run the new test and the existing attach-related Console tests.

## Task 8: Separate Transcript Title From Tab Controls Without Adding Height

**Addresses:** Item 8.

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: both Console CSS files
- Test: `Tests/UI/test_console_internals_decomposition.py`

- [ ] **Step 1: Write failing title/tab hierarchy test**

Add `test_console_transcript_header_and_tabs_have_distinct_visual_roles`.

Expected behavior:
- `#console-transcript-title` remains height 1.
- `#console-native-tab-strip` remains height 1.
- Title and tab strip have different classes or styles, for example title muted and active tab selected.
- No extra vertical row is introduced.

- [ ] **Step 2: Add header role classes**

In `console_session_surface.py`, set clear classes:
- `console-transcript-title`
- `console-session-tab-strip`

- [ ] **Step 3: Update CSS**

In both CSS files:
- Make title lower emphasis than selected tab.
- Give active tab a clear selected background.
- Keep row heights fixed at 1.

- [ ] **Step 4: Verify Task 8**

Run the new test and inspect screenshot at 1536 and 1920 widths.

## Task 9: Correct Composer Button Priority and Disabled States

**Addresses:** Item 9.

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: both Console CSS files
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing button priority tests**

Add tests:
- `test_console_composer_send_is_primary_only_with_draft`
- `test_console_composer_stop_is_disabled_when_idle`
- `test_console_composer_save_chatbook_is_secondary`

Expected behavior:
- Empty draft: Send is disabled or visually subdued.
- Non-empty draft: Send has primary class/state.
- Idle run: Stop is disabled or visually subdued.
- Active run: Stop becomes available.
- Save Chatbook uses a secondary class and does not compete with Send.

- [ ] **Step 2: Add composer action state API**

In `ConsoleComposerBar`, add:

```python
def sync_action_state(self, *, has_draft: bool, run_active: bool, can_save_chatbook: bool) -> None:
```

This method updates button `disabled` flags and classes.

- [ ] **Step 3: Call action sync from `ChatScreen`**

In `_sync_console_control_bar` or a dedicated helper, query the composer and call `sync_action_state` using:
- `bool(composer.draft_text().strip())`
- current `ConsoleChatController.run_state`
- existing `can_save_chatbook` computation

- [ ] **Step 4: Update composer methods**

After every draft mutation, call the action-state sync path or emit a local refresh hook so Send state updates immediately while typing.

- [ ] **Step 5: Verify Task 9**

Run:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected: PASS.

## Final Verification

- [ ] Run focused Console suites:

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

- [ ] Run style check:

```bash
git diff --check
```

- [ ] Regenerate screenshot QA:

```bash
.venv/bin/python /private/tmp/console_rails_batch_capture.py
```

- [ ] Inspect at least:
  - `Docs/superpowers/qa/console-persistent-rails/desktop-1920x1080/first-start-left-open-right-collapsed.png`
  - `Docs/superpowers/qa/console-persistent-rails/desktop-1920x1080/after-typing-guidance-dismissed.png`
  - `Docs/superpowers/qa/console-persistent-rails/desktop-1920x1080/both-rails-collapsed.png`
  - `Docs/superpowers/qa/console-persistent-rails/desktop-1536x900/first-start-left-open-right-collapsed.png`

- [ ] Self-review checklist:
  - No new static instructional block occupies transcript vertical space.
  - Composer is the clearest action target.
  - Collapsed rail controls are discoverable from screenshot alone.
  - Button priority matches idle, draft, and running states.
  - The screen keeps terminal-native density without excessive nested boxes.

- [ ] Commit:

```bash
git add tldw_chatbook Tests/UI Docs/superpowers/qa/console-persistent-rails
git commit -m "Refine console screen hierarchy and controls"
```
