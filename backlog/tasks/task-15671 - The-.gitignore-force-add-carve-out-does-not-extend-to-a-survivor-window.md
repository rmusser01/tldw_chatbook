---
id: TASK-15671
title: The .gitignore force-add carve-out does not extend to a survivor window
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-11 21:30'
updated_date: '2026-08-27 04:03'
labels:
  - console
  - change-review
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change tracking force-adds paths an agent touched even when .gitignore would hide them, so an agent writing to an ignored path (`.env` is the canonical case) still shows up for review. PR 3a-1 Task 6c's survivor window passes no `touched_paths` when it closes, so that carve-out does not apply to changes a sub-agent makes AFTER its turn: an ignored path written by a survivor surfaces inside its own turn's window and not after it. Closing this requires carrying a bounded projection of attributed child WRITE paths across the existing exact Git boundaries, including pending and inherited survivors; the gap is named in the code at `_close_post_turn_change_window` rather than half-built.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A survivor writing to a .gitignore'd path has that write surfaced in its own survivor window
- [ ] #2 The turn window's existing force-add behaviour is unchanged
- [ ] #3 A test writes an ignored path from a post-turn child and fails when the carve-out is absent
- [ ] #4 The gap comment at _close_post_turn_change_window is removed rather than reworded
<!-- AC:END -->

## Implementation Plan

1. Add a production-shaped RED regression that executes the real `write_file`
   tool from a child only after its parent turn returns and proves a new ignored
   file is absent from survivor review on current `dev`.
2. Extend `ShadowRepo`/`ChangeTurnTracker` so recorded eligible WRITE paths can
   be force-added atomically at baseline and fresh-end snapshots, while a
   supplied-SHA close primes the next exact snapshot without rewriting history.
3. Retain one bounded bridge-local path state per spawning turn, including
   pending, inherited, and E-in-flight children, and pass those paths through
   the existing parent/survivor lifecycle.
4. Claim successor B before it starts and serialize competing close callers so
   survivor and successor windows remain exact, abutting, and non-overlapping.
5. Run focused Git-backed lifecycle, boundary, ordinary-force-add, and failure
   tests; remove the named gap; then record verification and close the task.

ADR required: yes

ADR path: `backlog/decisions/092-console-live-child-write-path-boundaries.md`

Reason: the fix changes the cross-module baseline input and supplied-SHA
shadow-index semantics while preserving ADR-089's user-visible ownership.

Approved design:
`Docs/superpowers/specs/2026-08-26-task-15671-ignored-survivor-write-tracking-design.md`

Detailed implementation plan:
`Docs/superpowers/plans/2026-08-26-task-15671-ignored-survivor-write-tracking-implementation.md`
