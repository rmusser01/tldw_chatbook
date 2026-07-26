---
id: TASK-646
title: Complete destination handoff ownership and ACP target recovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 13:37'
updated_date: '2026-07-26 15:02'
labels:
  - architecture
  - state
  - reliability
dependencies:
  - TASK-645
references:
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-application-session-state-ownership-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining Study, Artifacts, and ACP navigation handoffs to the revisioned memory-only owner, repair the missing ACP target consumer, remove the dead Notes slot, and close the application-level ownership boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typed Study, Artifacts, and ACP slots preserve revisioned single-slot replacement semantics
- [ ] #2 Artifacts resolves only an exact canonical local:chatbook target, never substitutes the latest first-page record, releases transient failures for later lifecycle or user retry, and acknowledges success or terminal missing-target outcomes
- [ ] #3 ACP consumes its canonical local:acp_session record target by reconstructing the same identifier from the current runtime session; an exact match keeps that row selected and exposes the mounted detail pane, while malformed, stale, or unsupported targets receive explicit recovery
- [ ] #4 All listed raw application pending fields and the dead pending_notes_workspace_context slot are removed and an AST ownership guard prevents their return
- [ ] #5 Focused exact-target, replacement-race, privacy-sentinel, mounted-flow, and static checks plus the installed-wheel gate, product-maturity UI sentinel, and full suite pass after the integrated tranche
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/026-application-session-state-ownership.md
Reason: ADR-026 defines the remaining destination ownership and exact target recovery contract.
Full plan: Docs/superpowers/plans/2026-07-26-task-646-destination-handoffs.md

1. Extend the owner with Study, Artifact, and ACP channels and migrate producers.
2. Settle Study scope and section independently after restore.
3. Resolve Artifact targets through exact get_chatbook lookup with app-thread generation, restart, cancellation, and unmount settlement guards.
4. Complete current-only ACP session target recovery.
5. Remove every raw pending field and close the AST/privacy boundary.
6. Run focused, product-maturity (including the Phase 1 harness), installed-wheel, static, and full-suite gates before reconciling TASK-643 through TASK-646 together.
<!-- SECTION:PLAN:END -->
