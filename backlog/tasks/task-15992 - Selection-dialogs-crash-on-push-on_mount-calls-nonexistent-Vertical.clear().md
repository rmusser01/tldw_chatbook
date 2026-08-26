---
id: TASK-15992
title: 'Selection dialogs crash on push: on_mount calls nonexistent Vertical.clear()'
status: Done
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both the Note and Conversation selection dialogs call `Vertical.clear()` in `on_mount`; no such method exists on Textual containers (`remove_children` is the idiom), so pushing either dialog still ends in AttributeError from the dialog's own code. This is the THIRD pre-existing defect in these dialogs: TASK-15450 fixed their invalid `font-size: 10` (which poisoned the whole app stylesheet) and deliberately left this one un-papered-over — the mounted test in `Tests/UI/test_widget_css_consolidation.py` documents it and currently tolerates the AttributeError via a scope fence. Fixing this should also let that test drop its `"clear" not in str(raised)` escape hatch (see TASK-15994). Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both selection dialogs push and mount without raising
- [x] #2 The consolidation test's AttributeError tolerance is removed (the mounted pin asserts a clean open)
- [x] #3 Born-red evidence for the fix (test fails on the current dev behavior)
- [x] #4 Each dialog is driven once beyond mount (item selected, dismissal result asserted); any second defect found is fixed if small or filed
<!-- AC:END -->

## Implementation Plan

1. Remove the `"clear" not in str(raised)` escape hatch from
   `test_selection_dialog_opens_without_a_stylesheet_error` in
   `Tests/UI/test_widget_css_consolidation.py` (and its docstring paragraph)
   so a clean open is asserted.
2. Born-red: run the fence-removed test against the unfixed dialogs — it must
   fail with the `AttributeError: ... 'clear'` — capture the output.
3. Replace `Vertical.clear()` with `remove_children()` in
   `Widgets/Note_Widgets/note_selection_dialog.py` (`load_notes`) and
   `Widgets/conversation_selection_dialog.py` (`load_conversations`); re-run
   the mounted test green.
4. Drive each dialog beyond mount in a new `Tests/UI/test_selection_dialogs.py`:
   select an item, press Generate, assert the dismissal result. Suspicion to
   verify: the conversation dialog nests its per-item radios inside layout
   containers inside a `RadioSet`, and Textual's `RadioSet.pressed_index` does
   `self._nodes.index(...)` on direct children only — selecting should raise
   ValueError (second defect). If confirmed and small, fix (drop the outer
   RadioSet and enforce exclusivity in the dialog's own handler); born-red the
   driving test first either way.
5. ruff check + format touched files; run the consolidation test module and the
   new test file to completion with output captured to a file.

## Implementation Notes

Fixed the `.clear()` crash in both dialogs, removed the test's escape hatch,
and — driving the dialogs beyond mount — found and fixed a second defect that
made the conversation dialog unusable even after the crash fix.

- **`Vertical.clear()` → `remove_children()`** in
  `tldw_chatbook/Widgets/Note_Widgets/note_selection_dialog.py::load_notes` and
  `tldw_chatbook/Widgets/conversation_selection_dialog.py::load_conversations`.
  Intent of both `on_mount` paths is clear-then-repopulate (so `load_*` can be
  re-called); `remove_children()` snapshots current children, so the unawaited
  call followed by fresh `mount()`s is safe.
- **Fence removed**: `test_selection_dialog_opens_without_a_stylesheet_error`
  (`Tests/UI/test_widget_css_consolidation.py`) now re-raises ANY non-CSS
  exception unconditionally; the `"clear" not in str(raised)` hatch and its
  docstring paragraph are gone. Born-red: with the fence removed and the
  dialogs unfixed, both parametrizations failed with
  `AttributeError: 'Vertical' object has no attribute 'clear'`
  (note_selection_dialog.py:176, conversation_selection_dialog.py:213); green
  after the one-line fixes.
- **Second defect (found via AC #4, fixed here)**: the conversation dialog
  wrapped its per-item radios' container in a `RadioSet`
  (`#conversations-radio-set`). Textual's RadioSet only manages *direct*
  RadioButton children: selecting a nested radio crashed the RadioSet's
  message pump (`ValueError: RadioButton(...) is not in list` from
  `pressed_index` during `Changed` construction) AND its handler `event.stop()`ed
  the `RadioButton.Changed` before the dialog's own handler, so
  `selected_conversation_id` stayed `None` and Generate stayed disabled
  forever — the dialog could never return a result. Fix: dropped the outer
  RadioSet (comment in `compose()` explains why) and made
  `on_radio_button_changed` enforce one-of-N itself (turns the previous radio
  off under `self.prevent(RadioButton.Changed)`, mirroring RadioSet's own
  idiom; toggling the selected radio off clears the selection and disables
  Generate). Born-red first: both conversation driving tests failed with the
  ValueError/never-enabled symptoms before the fix.
- **New tests**: `Tests/UI/test_selection_dialogs.py` drives each dialog on
  `ConsolidatedCSSApp`. The note dialog is driven as a user would (a pilot
  click on a checkbox works: count label updates, Generate enables,
  dismissal returns `[note_id]`). The conversation dialog is driven
  PROGRAMMATICALLY — `query_one(...).value = True`, which bypasses layout,
  hit-testing and focus — because a `pilot.click` drive fails today: the
  dialog's own layout CSS collapses the list container to 2 rows and stacks
  both items on identical coordinates outside the clip, so a mouse click
  lands on the options section instead of a radio (filed as TASK-16470).
  Selection, exclusivity, untoggle, and the dismissal options-dict are
  asserted at the message/handler level.
- Verification: `Tests/UI/test_selection_dialogs.py` +
  `Tests/UI/test_widget_css_consolidation.py` → 18 passed. ruff check clean,
  ruff format applied on the four touched files. No User_Guide page covers
  these dialogs (searched), so no doc stamp needed. The measured visual delta
  from removing the RadioSet wrapper is nil: the TASK-15992 review rendered
  both structures and the list region draws empty either way, because of the
  pre-existing layout collapse now tracked as TASK-16470.
- **Scope honesty / follow-ups filed**: these dialogs are currently
  UNREACHABLE in production — both STTS import paths import four DB helpers
  that do not exist in `DB/ChaChaNotes_DB.py`, and a broad `except Exception`
  swallows the ImportError into a toast before `push_screen` ever runs
  (TASK-16471) — so this fix has no user-facing value until that lands. The
  conversation dialog's collapsed/unclickable list is TASK-16470. The same
  `.clear()`-on-a-container bug class survives at two currently-unreachable
  sites in `Widgets/embedding_template_selector.py` (TASK-16472).
