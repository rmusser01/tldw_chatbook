---
id: TASK-3790
title: >-
  Console: a key batched behind the slash misses the command popup (DraftChanged
  ordering)
status: To Do
assignee: []
created_date: '2026-08-08 23:36'
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
- [ ] #1 A key delivered in the same driver read as the slash that opens the command popup is handled against the popup's post-edit state, as it was before TASK-3749
- [ ] #2 The strict xfail on test_an_arrow_queued_behind_the_slash_still_navigates_the_popup (Tests/UI/test_console_composer_draft_changed.py) is removed and the test passes
- [ ] #3 The fix does not reintroduce a screen-to-composer callback inside on_key as the primary notification path
<!-- AC:END -->
