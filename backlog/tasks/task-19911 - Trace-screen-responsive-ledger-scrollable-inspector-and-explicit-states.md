---
id: TASK-19911
title: Trace screen responsive ledger scrollable inspector and explicit states
status: Done
assignee: []
created_date: '2026-08-22 18:29'
updated_date: '2026-08-23 04:58'
labels: []
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - Docs/superpowers/plans/2026-08-22-task-19911-19912-trace-v2-interface.md
  - >-
    backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Trace usable for first-time, keyboard, accessibility, and narrow-terminal users while adopting Trace as the canonical Console label.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The inspector exposes every rendered field through independent keyboard scrolling and a full-pane detail mode
- [x] #2 The ledger uses verified responsive tiers at 60x18, 80x24, 100x30, and 120x35 without hiding record identity or requiring horizontal scrolling for primary facts
- [x] #3 Live following, paused, imported, loading, incomplete, filtered, empty, and failure states are explicitly visible and actionable
- [x] #4 Trace is the canonical user-facing name and record kinds are humanized
- [x] #5 Pilot geometry and compositor tests prove reachability and responsive behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED Pilot/compositor tests at 60x18, 80x24, 100x30, and 120x35 using the production screen hierarchy and bundled styles; assert primary facts remain visible without horizontal scrolling and inspector focus is compositor-visible.\n2. Add RED state-contract tests for live following, paused, imported, loading, incomplete, filtered-empty, empty, and failure/retry states.\n3. Replace the clipped Static inspector with an independently focusable native vertical scroll owner, a visible overflow cue, and a full-pane detail mode while preserving stable selection.\n4. Rebuild ledger columns only when crossing responsive width tiers; keep record identity, humanized event, summary, and state at every tier, with metrics progressively disclosed.\n5. Adopt Trace as the canonical user-facing label, humanize record kinds centrally, add explicit state/filter summaries, and update the feature guide.\n6. Run focused UI/live/import tests, one batched four-viewport compositor inspection plus at most one correction pass, the Impeccable detector once after UI changes finish, Ruff, and diff checks.\n7. Complete independent specification and quality reviews, resolve findings with focused RED/GREEN evidence, then close the task.\n\nADR required: no.\nADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md\nReason: this directly implements the responsive, reachable, state-explicit Trace presentation already approved by ADR-080 without changing storage, ownership, or service boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the responsive, keyboard-complete Trace screen on the current dev base. Replaced the clipped Static details block with a focusable native VerticalScroll, truthful overflow cue, reversible full-pane mode, generic payload rendering, stable selection/legacy collision identity, revision-safe live refresh and single-flight retry behavior. Added responsive ledger tiers with primary identity/event/summary/state at 60-99 columns, Tokens/Duration at 100-119, and full metrics at 120+, all without horizontal overflow. Added orthogonal live/imported/incomplete/loading/filter/empty/failure states, truthful clear-filter recovery, canonical Trace copy, and human labels; updated the feature guide and Console shortcut copy.

Verification: production three-stylesheet Pilot/compositor coverage at 60x18, 80x24, 100x30, and 120x35; one batched visual pass plus one correction pass; Impeccable detector returned []; two independent final reviews returned Ready: Yes. After merging origin/dev d60ebe1d0, the Trace UI suite passed 122 tests and projection/DB/lineage smoke passed 85 tests with 2 log-dependent tests deselected because pristine latest dev has a pre-existing RAG_Search.config_profiles circular import that disables the run-log writer. Ruff, format, and diff checks passed.

ADR required: no new ADR. ADR-080 already governs the responsive/reachable Trace presentation and no storage or service boundary changed. No new lesson entry: the production-CSS/compositor and focus-visible traps are already documented in lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
