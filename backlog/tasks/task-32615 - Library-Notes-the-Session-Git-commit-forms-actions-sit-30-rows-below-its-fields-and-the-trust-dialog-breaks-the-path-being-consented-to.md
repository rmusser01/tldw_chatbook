---
id: TASK-32615
title: >-
  Library Notes: the Session Git commit form's actions sit 30 rows below its
  fields and the trust dialog breaks the path being consented to
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-15 06:41'
updated_date: '2026-09-15 18:32'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D9 and the layout half of D2, personas Alex and Sam, Obsidian workflow.

What happened. Session Git's consent design is one of the best things on the screen -- the commit review states What / Where / Impact / Recovery, names the identity, the branch and parent, says hooks will not run and the commit will be unsigned, and promises no unrelated staged content will be committed (B cap 44, verified end to end against git log). Its geometry undoes some of that.
- Commit form: Subject and Body at rows 17-27, the only actions (Cancel commit, Review commit) at row 48, twenty blank rows between, no more-below cue (B caps 42, 43). Same for the commit review (B cap 44).
- Repository-trust modal: rendered inside the roughly 45-column right pane, so the repository path wraps as '/Users/.../crit4/B/power/vaul' plus 't' on the next line, and the status truncates to 'Status: TRUST REQU…' (B cap 39). A security consent dialog must never mangle the identifier being consented to.
- And after a successful commit the message 'Committed 1 session note as c1db79e…; unrelated changes untouched.' renders under the Danger heading at 100x30 and 60x24, with 'Commit review ready.' doing the same at 235x52 (B caps 44, 47, 48, D12).

Cause INFERRED for all three (not traced).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The commit form's actions sit within the reader's view of its fields, or the pane states that more follows
- [x] #2 The repository-trust dialog renders the full path it asks consent for, without wrapping it mid-token, and shows its status in full
- [x] #3 A success message never renders under a Danger heading
- [x] #4 The three sizes are captured after the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix the path rendering in the trust dialog first -- a consent dialog mangling its own identifier is a correctness bug, not a layout one.
2. Measure the commit form's field-to-action gap, and close it without pushing the actions off a short pane.
3. Find why a success receipt paints under Danger and move it, not the heading.
4. Capture all three sizes after.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three findings, measured through the panel harness and the production Library harness at 235x52, 100x30 and 60x24.

**The trust dialog (taken first, as a correctness bug).** `ConfirmationDialog` gives its message 54 cells (a fixed 60-cell container less one border cell either side and `padding: 1 2`), and Textual folds a token that does not fit at whatever column it runs out of. Measured: '/Users/macbook-dev/Documents/qa/library-notes-and-vaults/' painted as '...library-notes-and-vaul' + 'ts/crit4/power/vault' -- the assessor's own '...vaul' + 't'. The path may not be ELIDED here either: it is the thing consent is being given for. So it is broken at its own separators instead (`fold_path_lines` in `Utils/Utils.py`, beside `elide_path_middle`), on its own line under 'Repository:' so it gets the full 54 cells. Every character survives and no component is split. Folded AFTER `_repository_path_for_display`, never before -- that pass turns a real newline into a literal '\\n' and would escape the fold's own breaks back out.

The AC's other half -- 'shows its status in full' -- was NOT reproduced: `render_untrusted` already sets the status with `complete=True`, which gives it `-complete-copy` (`max-height: 100%`, `text-wrap: wrap`) and skips `_fit_two_line_copy` entirely. Measured at a 45-cell pane it paints 'Status: TRUST REQUIRED — Trust this repository to check current session notes.' in full across two rows. Ticked as already true rather than 'fixed'.

**The commit form.** `#file-notes-git-commit-body` is a `1fr` scroll, so it took every spare row of the pane and the footer sat at the floor: measured 40 rows between the form's last field and Cancel/Review at 235x52. Same defect `_sync_row_list_height` fixed one surface over, and the same shape of fix: the `1fr` stays (it is what lets the scroll shrink and keeps the footer on screen), and the ceiling comes from the visible phase's own settled height, which Textual leaves at its content height whatever the scroll is capped to. Deferred one refresh because `_show_commit_phase` has only just switched which phase is displayed. A negative control at 40x20 pins that a phase taller than the pane still leaves the footer painted on screen -- the exact regression `_sync_row_list_height`'s comment warns about.

**The receipt under Danger.** `#file-notes-action-status` was the LAST child of `#file-notes-editor-pane`, below `#file-notes-manage-region` whose final heading is 'Danger', so every `_set_action_status` receipt -- 'Committed 1 session note as …', 'Commit review ready.' -- painted under it. Moved to directly under the save state, above both regions. One compose-order change; no copy changed.

**Modified:** `Widgets/Library/library_file_notes_git_panel.py` (`_sync_commit_body_height`, `SessionGitTrustDialog`, `_TRUST_DIALOG_MESSAGE_WIDTH`), `Utils/Utils.py` (`fold_path_lines`), `Widgets/Library/library_file_notes_workspace.py` (compose order only), `Tests/UI/test_library_notes_w5_session_git_layout.py` (new), `Docs/User_Guide/library/file-notes.md`.

**Red first:** '40 rows between the commit fields and their actions'; "line '/Users/macbook-dev/Documents/qa/library-notes-and-vaul' breaks a path component in half"; receipt-under-Danger failed at all three sizes. The trust-dialog pin needed a second attempt -- the first path I chose folded at a separator by luck and passed unfixed, which is recorded in the test.
<!-- SECTION:NOTES:END -->
