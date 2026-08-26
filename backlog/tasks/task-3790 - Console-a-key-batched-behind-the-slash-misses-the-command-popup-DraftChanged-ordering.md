---
id: TASK-3790
title: >-
  Console: a key batched behind the slash misses the command popup (DraftChanged
  ordering)
status: Done
assignee: []
created_date: '2026-08-08 23:36'
updated_date: '2026-08-09 02:34'
labels:
  - console
  - refactor
  - defect
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3749 replaced the screen's after-the-edit callbacks with a ConsoleComposerBar.DraftChanged message. Textual delivers that message through the pump, so when two keystrokes arrive in a SINGLE driver read (a key macro, a text expander, tmux send-keys -- human typing is orders of magnitude too slow), the second key is handled before the screen has opened the slash-command popup, and that key is ignored. Measured as a two-arm A/B: batched slash+Down used to highlight the second entry and now ignores the Down; batched slash+Enter used to accept the highlighted suggestion and now ignores the Enter. In both cases the popup still opens, the draft is untouched, nothing is sent, and the next keypress behaves normally -- so this is a lost keystroke under programmatic input, not a data-loss or wrong-send bug. It matters mainly because this repo drives live TUI verification with tmux send-keys.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A key delivered in the same driver read as the slash that opens the command popup is handled against the popup's post-edit state, as it was before TASK-3749
- [x] #2 The strict xfail on test_an_arrow_queued_behind_the_slash_still_navigates_the_popup (Tests/UI/test_console_composer_draft_changed.py) is removed and the test passes
- [x] #3 The fix does not reintroduce a screen-to-composer callback inside on_key as the primary notification path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on the task-3749 branch itself rather than deferred. The gap: `DraftChanged`
delivers through the message pump, so a key queued behind the slash in the same
driver read consulted a popup whose sync had not run. Fix, two halves, both
mutation-verified:

1. `_ensure_console_command_popup_current` at the on_key routing point,
   **gated on the synced draft TEXT** -- an Escape-dismissed popup (no edit, no
   text movement) stays dismissed; removing the gate fails
   `test_escape_dismissal_survives_a_following_arrow_key`.
2. `show_suggestions` is idempotent for an identical suggestion list, so the
   deferred `DraftChanged` re-sync cannot yank the highlight a routed Down just
   moved; removing that fails
   `test_deferred_draft_changed_does_not_yank_the_moved_highlight`.

The first gate attempt used the composer's `_draft_generation` and no-oped on
every keystroke: that counter is an undo-checkpoint marker `insert_text` never
advances -- measured with an on_key entry probe, not assumed. The text compare
is the honest signal because the sync is a pure function of the draft.
<!-- SECTION:NOTES:END -->
