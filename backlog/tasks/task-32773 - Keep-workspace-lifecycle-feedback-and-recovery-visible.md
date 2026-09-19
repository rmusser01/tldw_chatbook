---
id: TASK-32773
title: Keep workspace lifecycle feedback and recovery visible
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 06:12'
updated_date: '2026-09-18 06:47'
labels:
  - ui
  - settings
  - design-system
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace creation, rename, archive and restore must preserve visible keyboard context and expose truthful outcomes and recovery across compact and wide layouts. Review the mounted workflows and repair confirmed gaps while preserving existing lifecycle authority and cancellation contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create validation and partial folder recovery retain entered values and expose the complete error beside a reachable keyboard action.
- [x] #2 Rename, activation, archive, Undo and Restore as keep visible focus and truthful local outcomes, including duplicate names and literal workspace labels.
- [x] #3 Archive and restore retain existing refusal, cancellation, delayed completion, receipt ownership and active-workspace semantics.
- [x] #4 Targeted regressions and dark/light compact/wide native journeys verify real private persistence, cancellation, recovery and clean shutdown.
- [x] #5 New Workspace labels and recovery text remain readable on the current dark or light theme
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md (existing), plus ADR-150/161 for presentation. Reason: verify and repair existing lifecycle presentation without changing storage or authority. 1. Read shared create and Settings lifecycle implementations, prior tasks and targeted tests; preserve failing mounted keyboard/painted-state probes before production changes. 2. Repair only confirmed visible-feedback, draft and focus gaps using existing component/token patterns; preserve async storage and receipt ownership. 3. Exercise Create validation/cancel/partial binding retry, Rename, Set active, Archive cancel/confirm, Undo, name-conflict Restore as and delayed publication; run affected tests/static/governance with independent review. 4. Verify real private-profile native compact/wide dark/light journeys, inspect final captures and lifecycle, update the completion ledger and draft PR. Allocation: refreshed origin; all reachable task paths plus 30 worktrees show maximum32772; candidate32773 has no task references across280 branch/remote/tag refs and all worktrees.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Create, Rename and Restore now keep local feedback visible; partial folder recovery preserves the created identity and offers Retry folders or Keep workspace with already-saved fields locked. Activation, Undo and Restore retain useful replacement focus, archived rows wrap, and archive text preserves literal names. Native inspection also found and repaired the black light-theme Create surface using ADR-150/161 app tokens and matching standalone theme variables. ADR-147 storage/refusal/cancellation/receipt ownership is unchanged; no new ADR. Final native run004 passed four cells with 20 inspected captures, real private registry writes and clean shutdown; evidence: Docs/superpowers/qa/2026-09-17-settings-workspace-lifecycle/README.md. Existing tests were isolated before imports and reveal controls before activation; original assertions retained. The CSS guard was aligned to the already-saved TASK-32765 list-introduction height; Library production files unchanged. Independent review and source/capture hashes verified. 172 distinct cases pass across final and corrective runs: 13 new UI cases, 24 existing Create cases and 135 related cases (the batch passed 127 and its eight old pane-harness failures pass after repair in the separate nine-case run). A final 13-case token/build guard also passes and overlaps the count. Scoped Ruff/format checks and baseline diagnostic comparison pass; all 24 existing Create bodies retain their assertions and only replace below-viewport click calls with the visible-click helper. No full suite or provider requests.
<!-- SECTION:NOTES:END -->
