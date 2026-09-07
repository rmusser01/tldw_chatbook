---
id: TASK-1977
title: 'Change review: auto-register nested repos as tracked sub-roots (fast-follow)'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
dependencies:
  - TASK-1976
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the nested-repo hole: repos detected inside a root become their own tracked sub-roots — own shadow repos keyed by their canonical paths, excluded from the parent's shadow repo, bounded depth. The review aggregates parent + sub-root diffs per turn; the TASK-1976 banner disappears for auto-registered children.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An edit inside a nested repo now appears in the turn's review, attributed to the sub-root
- [x] #2 The parent's shadow repo excludes the child (no gitlink churn rows)
- [x] #3 Sub-root discovery is bounded (depth/count) with disclosure when the bound truncates
- [x] #4 Removing the child repo un-registers its sub-root via existing GC
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. begin_turn expands each root's scan.nested_repos (first max_sub_roots,
   knob default 20) into additional tracked roots — each gets its own budget
   gate, B snapshot, and shadow repo; depth bound = 1 by construction (the
   scan stops descending at a nested repo; grandchildren stay disclosed via
   the CHILD's own banner when it scans as a root)
2. end_turn: parent's nested_repos disclosure = E-scan nested MINUS the
   auto-registered children (disclosure covers exactly the untracked);
   beyond-bound children remain disclosed (AC#3 truncation honesty)
3. Attribution (AC#1) falls out of per-(run,root) rows + the screen's
   existing multi-root labels; parent excludes child already (1976, AC#2);
   deleted child un-registers via the existing orphan GC (AC#4)
4. TDD against real git; sabotage first-try passes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`ChangeTurnTracker._baseline` now expands each ORIGINAL root's detected
nested repos (first `max_sub_roots`, knob default 20, flat-section +
env-overridable) into tracked roots of their own: each gets the same budget
gate, B snapshot, and its own shadow repo keyed by canonical path. Depth is
1 by construction — only the caller's original roots expand, so a
grandchild repo stays DISCLOSED via its parent's banner instead of
recursing; a cycle/dedupe guard (`seen`, resolved paths) protects the queue.

Disclosure now covers exactly what is untracked: a registered child leaves
`nested_repos` (both the baseline set and the E-scan set are filtered), a
beyond-bound child stays in it (AC#3 truncation honesty — asserted
order-agnostically since walk order isn't guaranteed), and a repo cloned
mid-turn is disclosed this turn and auto-registered the next. Attribution
(AC#1) falls out of per-(run,root) rows plus the screen's existing
multi-root labels; parent exclusion (AC#2) was already TASK-1976's exclude
+ pathspec work; a deleted child's shadow container goes through the
existing TASK-1975 orphan GC (AC#4, tested).

Contract updates to three TASK-1976 tests whose premises 1977 supersedes
(hole closed, not just disclosed): parent-diff purity is the surviving
core; the screen's banner test pins the disclosure path with the bound at
zero. Key test subtlety: a CLEAN sub-root correctly yields no record —
assertions must mutate the child to observe registration. Sabotage
(auto-registration disabled) failed 6 tests. 256 green across all
change-review suites before push.
<!-- SECTION:NOTES:END -->
