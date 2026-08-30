---
id: TASK-24529
title: Await pruning Console selection menu remounts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 00:59'
labels:
  - console
  - textual
  - reliability
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-29-console-selection-menu-remount-race-design.md
documentation:
  - Docs/superpowers/plans/2026-08-29-console-selection-stability.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent app-fatal duplicate selection-menu IDs when a completed Console text selection replaces a menu whose removal is already pending.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A completed text selection waits for every previously attached Console selection menu to detach, including a menu already marked for pruning, before mounting its replacement
- [ ] #2 Ordinary fire-and-forget menu dismissal retains its current non-pruning behavior without duplicate removal work
- [ ] #3 Immediate no-yield replacement leaves exactly one new non-pruning menu mounted, the previous menu detached, and the app running without `DuplicateIds`
- [ ] #4 Settled consecutive drags, menu placement, menu actions, feedback, focus, and dismissal behavior remain unchanged under focused Console tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a deterministic already-pruning no-yield remount regression.
2. Replace the remount boundary with the public awaited screen query while preserving ordinary dismissal.
3. Run focused Console selection-menu and dismissal verification.
4. Complete task evidence and self-review.

Detailed plan: Docs/superpowers/plans/2026-08-29-console-selection-stability.md
ADR required: no
ADR path: N/A
Reason: existing Textual lifecycle ordering only.
<!-- SECTION:PLAN:END -->
