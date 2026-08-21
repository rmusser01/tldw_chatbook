---
id: TASK-19050
title: Review-rail badge and strip window-scope mismatch
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - console
  - change-review
  - ux-polish
dependencies:
  - TASK-18060
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Inspector rail's "Changed files" section badges a file with a
`✎ N` note count derived per `(root, path)` across every window/run in
the conversation's history (`AgentRunsChangeReviewProvider.
conversation_changed_files`, TASK-18060 Task 2). But clicking a badged
row opens the Review screen pinned to the NEWEST `change_snapshots`
window for that path (`_open_change_review`'s `initial_run_id`/
`initial_path` recipe), while the notes strip inside that screen filters
strictly by the FOCUSED leaf's own `snapshot_id` (`_note_matches_leaf`,
Qodo #6 / PR #1779). When a run holds two windows on the same root+path
(the turn's own window and a surviving sub-agent's post-turn window,
TASK-18060 Task 2's own documented shape) and the note lives on the
OLDER window, the badge's count is honest but the landing spot is wrong:
the user sees `✎ 1` on the file, opens it, and the strip is empty --
the note is one `j`/`k` step away on the sibling window, with nothing on
screen pointing at it. Final-review Minor #4, 2026-08-20.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Clicking a badged file in the rail always lands the user somewhere the counted note is actually visible -- either by widening the initial selection to the specific window the note belongs to, badging per-window instead of per-(root, path), or giving the strip an explicit hint naming the sibling window that holds the note
- [ ] #2 The notes strip's existing snapshot-scoped filtering (a note only renders under its own window, never a sibling's) has no regression -- covered by a test asserting both windows' behavior together, not just the fixed click-through
<!-- AC:END -->
