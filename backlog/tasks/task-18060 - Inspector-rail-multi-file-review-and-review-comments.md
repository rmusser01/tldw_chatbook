---
id: TASK-18060
title: Inspector-rail multi-file review and review comments
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-18'
labels:
  - console
  - change-review
  - ux
dependencies:
  - TASK-16800
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Arc A of the V2 turn-file-card design, split out of TASK-16801 (owner
ruling 2026-08-18: tackle the two V2 halves individually; this half
first). Today a user reviews changes one card/turn at a time, and the
Review screen is reachable per turn only; nothing shows the
conversation's changed files across all turns, and review feedback is
limited to the card's hunk notes.

This task adds (per the code-grounded spec,
`Docs/superpowers/specs/2026-08-18-console-review-rail-design.md`):
a "Changed files" section in the existing Inspector rail listing the
conversation's cross-turn latest state per file (cached-summary pattern —
never a DB/git read on the rail's sync tick), click-through to the
existing Review screen focused on that file, and plannotator-style
commenting in the Review screen — a comment on a specific diff line or on
the whole file — anchored to the immutable snapshot diffs and delivered
to the agent through the same TASK-16800 auto-attach loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The Inspector rail shows a Changed-files section listing the conversation's changed files across ALL turns (latest state per file: status, ±counts, note badge), capped with an honest overflow tail, and rendering nothing when the conversation has no recorded changes
- [ ] #2 The section's data is never computed on the rail's sync tick: an unchanged conversation state performs no recompute (verified by test), and a new turn's changes appear via one off-thread refresh
- [ ] #3 Selecting a listed file opens the existing Review screen on that file's turn with that file focused (constructor-state plumbing; no post-push race)
- [ ] #4 In the Review screen, the user can attach a comment to a specific diff line (cursor over the rendered diff) and to the whole file, without leaving the screen; comments are validated and persisted like TASK-16800 notes
- [ ] #5 Line and file comments are delivered to the agent through the existing auto-attach loop with kind-aware block and disclosure rendering (byte-identical live vs resume), stamped by exact id, and surviving session resume
- [ ] #6 The Review screen displays the focused file's existing feedback (hunk notes, file comments, line comments) with pending-vs-sent state; pending comments can be deleted, delivered ones cannot
- [ ] #7 Revert behavior, the single-file diff render cap, and the `[console] turn_file_cards` kill-switch behavior are all unchanged (no regression to their existing pinned tests)
<!-- AC:END -->
