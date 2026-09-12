---
id: TASK-15671
title: The .gitignore force-add carve-out does not extend to a survivor window
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 21:30'
updated_date: '2026-08-27 21:24'
labels:
  - console
  - change-review
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change tracking force-adds paths an agent touched even when `.gitignore` would hide them, so an agent writing to an ignored path still shows up for review. The survivor window previously had no equivalent input, allowing an ignored path written after the parent turn to escape that survivor's review card. The fix carries a bounded projection of attributed child WRITE paths across the existing exact Git boundaries, including pending and inherited survivors, without adding durable storage or a filesystem watcher.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A survivor writing to a .gitignore'd path has that write surfaced in its own survivor window
- [x] #2 The turn window's existing force-add behaviour is unchanged
- [x] #3 A test writes an ignored path from a post-turn child and fails when the carve-out is absent
- [x] #4 The gap comment at _close_post_turn_change_window is removed rather than reworded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a production-shaped RED regression that executes the real `write_file`
   tool from a child only after its parent turn returns and proves a new ignored
   file is absent from survivor review on current `dev`.
2. Extend `ShadowRepo`/`ChangeTurnTracker` so recorded eligible WRITE paths can
   be force-added atomically at baseline and fresh-end snapshots, while a
   supplied-SHA close binds late paths to the claimed successor without
   rewriting history or leaving unowned index state.
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
successor-handoff semantics while preserving ADR-089's user-visible ownership.

Approved design:
`Docs/superpowers/specs/2026-08-26-task-15671-ignored-survivor-write-tracking-design.md`

Detailed implementation plan:
`Docs/superpowers/plans/2026-08-26-task-15671-ignored-survivor-write-tracking-implementation.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a bridge-local child WRITE-path projection covering pending, live,
  inherited, and E-in-flight children. State remains bounded to owner identity,
  scope counts, and normalized paths; no schema or file-content cache was
  introduced.
- Added an exact successor-boundary claim and shared close-completion handoff.
  Survivor E and successor B remain abutting, concurrent closers share one
  outcome, and a timeout fails change tracking closed instead of inventing an
  overlapping review window.
- Extended baseline and end snapshots to force-add eligible recorded paths
  atomically. Supplied-SHA closure preserves the supplied commit and binds late
  paths to the exact claimed successor handle; successor E stages them inside
  its own locked snapshot, so another conversation cannot consume shared index
  state. Exact paths use Git's native NUL-delimited `update-index --stdin`
  transport rather than custom argv chunking.
- Added production-shaped and race-focused coverage for ignored post-turn
  writes, pending and inherited children, B/E ownership, concurrent close,
  oversize/refusal behavior, and injected failures. Final review also added a
  pending-at-successor-B regression so later child writes receive the existing
  concurrent-sub-agent disclosure.
- After final review and rebasing onto current `origin/dev`,
  `Tests/Chat/test_change_turn_tracking.py` passed 90 tests and
  `Tests/Workspaces/test_change_tracking.py` plus
  `Tests/Workspaces/test_change_bounds.py` passed 62 tests. Scoped Ruff,
  `compileall`, and `git diff --check` also passed.
- Qodo's three findings were validated and fixed: force-path inputs now use
  the shared path validator with hidden paths explicitly allowed, tracking
  errors use one named length constant, and a cross-conversation regression
  proves supplied-boundary paths remain owned by their claimed successor.
- The adjacent Chat aggregate has two baseline-only failures in
  `test_child_run_scope_ordering.py` and `test_fleet_settle_fanout.py`: both
  monkeypatch `_persist` with the old `(self, run_id, outcome)` signature while
  current `origin/dev` calls `_persist(run_id, outcome, durable_handle_ids)`.
  The task-focused module does not fail.
- Two independent final reviewers passed the disclosure fix, path-transport
  simplification, state cleanup, acceptance criteria, and governance links.
- Reviewed the production diagnostic inventory drift reported by CI. The six
  net-new `console_agent_bridge.py` warning calls are static lifecycle and
  fail-closed diagnostics; none interpolates user content, secrets, paths, or
  URLs. Regenerated the pinned inventory after that statement-level review.
- Governance: [ADR-092](../decisions/092-console-live-child-write-path-boundaries.md),
  [design](../../Docs/superpowers/specs/2026-08-26-task-15671-ignored-survivor-write-tracking-design.md),
  and [implementation plan](../../Docs/superpowers/plans/2026-08-26-task-15671-ignored-survivor-write-tracking-implementation.md).
<!-- SECTION:NOTES:END -->
