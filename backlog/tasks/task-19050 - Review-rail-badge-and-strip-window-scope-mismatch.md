---
id: TASK-19050
title: Review-rail badge and strip window-scope mismatch
status: Done
assignee: []
created_date: '2026-08-20'
updated_date: '2026-08-26 18:04'
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
- [x] #1 The duplicate Inspector Changed files surface is retired; per-turn cards and Review remain available
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Supersede the mismatched Inspector projection by removing that duplicate surface under TASK-22305 and ADR-089 while preserving per-turn cards and Review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Superseded by TASK-22305 and ADR-089. The rail surface that created the badge/window mismatch was removed; per-turn cards now own inline changed-file review and safe Undo All, while ambiguous same-root runs open Review.
<!-- SECTION:NOTES:END -->
