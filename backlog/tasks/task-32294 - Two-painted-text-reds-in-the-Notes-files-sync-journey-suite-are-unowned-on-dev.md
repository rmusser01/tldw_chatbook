---
id: TASK-32294
title: >-
  Two painted-text reds in the Notes files-sync journey suite are unowned on
  dev
status: To Do
assignee: []
created_date: '2026-09-10 12:55'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - file-notes
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_notes_files_sync_journey.py` has two tests that are red
on clean `dev` and belong to no open task, so nobody is watching them: a
reader who runs that file while working on Library ▸ Notes cannot tell a
regression they just caused from the reds that were already there. Both
assert on painted text or on what the compositor actually shows, so both are
either a real paint regression or a stale expectation — which of the two is
the work.

Found while landing PR #2557 (wave-2 Notes fix wave), where they had to be
confirmed as pre-existing before the PR's own results could be read. They are
not caused by that PR: the same two names, with byte-identical assertion
output, fail on the branch and on the `dev` tip it was merged from.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `test_lasting_setup_keeps_server_unavailable_copy_painted[size0]`
  passes, or the assertion is corrected to what the screen is supposed to
  paint with the reason recorded
- [ ] #2 `test_folder_files_and_session_git_use_supported_40x20_navigator`
  passes, or the assertion is corrected to what a 40x20 terminal is supposed
  to show with the reason recorded
- [ ] #3 Whichever of the two turns out to be a product defect rather than a
  stale expectation is fixed at its source, not by relaxing the assertion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence recorded at filing time (not yet investigated).

Baseline: clean `dev` at 068535986e, in a detached worktree, `pytest -q
-p no:randomly` on the two node ids -> `2 failed, 1 passed`. The `[size1]`
(60x20) parametrisation of the first test passes; only `[size0]` (120x36)
fails.

1. `test_lasting_setup_keeps_server_unavailable_copy_painted[size0]`
   (file line 650 on the PR #2557 branch, 620 on dev). The widget-level
   assertions above it pass — `#notes-sync-destination-server` is disabled
   and its reason renderable does contain "server sync-folder capability not
   installed" — but the screen does not paint it:

   ```
   assert "capability not" in painted
   AssertionError: assert 'capability not' in '                     ╭────────────╮ ...
   enter run action | esc back to notes | f1 help · f6 next pane · ctrl+p palette · ctrl+q quit  '
   ```

   The box-drawing characters at the head of the painted snapshot suggest a
   modal or overlay is on screen when the snapshot is taken, i.e. the
   `scroll_to_widget` before it did not put the reason in view. Note the
   preceding `assert "server sync-folder" in painted` passes, so only the
   later half of the copy is missing — consistent with clipping rather than
   with the reason being absent.

2. `test_folder_files_and_session_git_use_supported_40x20_navigator`
   (file line 1525 on the branch, 1504 on dev). At 40x20 the Session Git
   button is mounted but not composited:

   ```
   assert session_git in pilot.app.screen._compositor.visible_widgets
   AssertionError: assert Button(id='file-notes-session-changes',
   classes='-textual-compact -style-default') in {Static(id='footer-key-quit'):
   ..., Static(...nav-overflow-hint -style-default'): ..., ...}
   ```

   The compositor at that size reports `Size(width=40, height=20)` and does
   hold `library-notes-source-strip` and `library-header-line`, so the shell
   is up; it is the authority/source row's own contents that do not fit or do
   not paint. The test's earlier `"Folder files" in _painted_text` assertion
   passes.

Both were reproduced twice: on `dev` 068535986e and on
`fix/library-notes-r-file-notes` after that dev was merged in, with identical
assertion text. Neither name appears in any open task at filing time.
<!-- SECTION:NOTES:END -->
