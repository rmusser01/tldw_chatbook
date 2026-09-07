---
id: TASK-20937
title: Make Console rails edge-native and organize conversations by workspace
status: In Progress
assignee: []
created_date: '2026-08-22 19:31'
labels:
  - console
  - ux
  - workspace
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md
  - >-
    Docs/superpowers/plans/2026-08-22-task-20937-console-edge-rails-workspace-tree.md
  - backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved Console redesign so Context and Inspect form the application edges, Context sections use section-specific natural-height ceilings, named-workspace conversations live in a native Tree, Default and unassigned conversations remain in a flat browser, and Character art uses a stable 35-row contain layout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Context and Inspect render as the application side edges below the unchanged full-width Console header, with no horizontal workspace inset or nested side frame and exactly one stable divider per rail/transcript boundary.
- [ ] #2 Open Context sections hug content through their approved 15-, 20-, or 35-row ceiling, expose a separate local hint only on overflow, and remain reachable through ordinary Context outer scrolling without shared-budget shrinking, `· no room`, or `[>]` reprioritization; Inspector remains fixed at 20.
- [ ] #3 Workspaces renders every visible/selectable non-retired named workspace as a top-level native Tree node with associated conversations as children, while Conversations contains only Default-workspace and unassigned records and no conversation appears in both projections.
- [ ] #4 Workspaces and Conversations searches are independent; full-scope Tree search, per-workspace paging, loading/error/Retry state, and stale/duplicate async-result rejection preserve focus, membership, and temporary disclosure semantics.
- [ ] #5 Starred is a property and action rather than a duplicate location; starred conversations sort first inside their one owner, remain operable from Tree and flat rows, and never appear in a cross-owner Starred aggregate or duplicate search result.
- [ ] #6 Workspaces preserves a compact pinned Switch/New/RAG strip with an explanatory RAG Scope tooltip, one-row active identity, a working Default route, a naturally sized Tree with at least eight visible rows when demand permits, literal-safe labels, ASCII glyph fallback, and the approved pointer/keyboard/focus behavior.
- [ ] #7 Character's complete body uses at most 35 initial content rows; valid images render fully with aspect-ratio-preserving contain behavior in the measured space left by controls, missing/corrupt images retain recovery copy, and one equality-bounded follow-up cannot oscillate.
- [ ] #8 Responsive rail intent, ADR-043 width floors, section-open and disclosure preferences, session-local offsets, nested-scroll handoff, focus recovery, active/run markers, conversation resume, and Inspector ownership/navigation do not regress.
- [ ] #9 Deterministic small/representative/stress measurements report median, p95, materialized node counts, and reconciliation counts; any representative median regression over 20% is corrected or explicitly accepted without an unsupported speed claim.
- [ ] #10 Focused changed-functionality tests, scoped Ruff/format checks, source/generated CSS integrity, production-compositor geometry, iTerm2 and Windows Terminal parity at equal rows/columns, user documentation, ADR-083, and every child task are complete before TASK-20937 is marked Done.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TASK-20937.1 parameterizes Context's 15/20/35 bounded bodies and retires shared-budget/no-room allocation while preserving Inspector at 20.
2. TASK-20937.2 splits exclusive workspace versus Default/unassigned projections, starred-first ordering, independent search lanes, and per-workspace page attempts.
3. TASK-20937.3 removes the inset workspace frame and establishes one dimensionally stable divider owner per rail.
4. TASK-20937.4 mounts the native Workspace Tree with compact pinned controls, literal labels, focus/key guards, paging actions, and keyed updates.
5. TASK-20937.5 measures Character controls and contains complete art within the remaining part of the 35-row body using one equality-bounded follow-up.
6. TASK-20937.6 runs the production CSS/performance/terminal evidence, updates user documentation, and closes every child plus this parent.

Detailed plan: `Docs/superpowers/plans/2026-08-22-task-20937-console-edge-rails-workspace-tree.md`.

ADR required: yes
ADR path: `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`
Reason: ADR-083 records the long-lived rail edge ownership, per-section vertical policy, exclusive workspace conversation ownership, native Tree/focus exception, split async lanes, and Character contain boundary.
<!-- SECTION:PLAN:END -->

## Closeout Status (2026-08-23)

The refreshed same-commit benchmark explicitly accepts all representative
diagnostic differences over 20%: the mounted/settled Tree path includes native
mount, deferred layout/paint, compositor rendering, and full-scope search nodes,
whereas the frozen baseline is projection-only. Representative medians are
48.020 ms initial mount, 22.668 ms marker update, 70.492 ms search apply/clear,
and 22.166 ms selection. These measurements do not support a relative speed
claim, and none is made.

TASK-20937 remains In Progress. The exact focused gate passes all 488 tests, but
neither the iTerm2 nor same-commit/equivalent-cell Windows Terminal operator
checklist and captures have been supplied. The exact remaining capture checklist
is recorded in TASK-20937.6.
