---
id: TASK-646
title: Complete destination handoff ownership and ACP target recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 13:37'
updated_date: '2026-07-26 21:35'
labels:
  - architecture
  - state
  - reliability
dependencies:
  - TASK-645
references:
  - backlog/decisions/033-application-session-state-ownership.md
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
- [x] #1 Typed Study, Artifacts, and ACP slots preserve revisioned single-slot replacement semantics
- [x] #2 Artifacts resolves only an exact canonical local:chatbook target, never substitutes the latest first-page record, releases transient failures for later lifecycle or user retry, and acknowledges success or terminal missing-target outcomes
- [x] #3 ACP consumes its canonical local:acp_session record target by reconstructing the same identifier from the current runtime session; an exact match keeps that row selected and exposes the mounted detail pane, while malformed, stale, or unsupported targets receive explicit recovery
- [x] #4 All listed raw application pending fields and the dead pending_notes_workspace_context slot are removed and an AST ownership guard prevents their return
- [x] #5 Focused exact-target, replacement-race, privacy-sentinel, mounted full-production-app, and static checks plus the installed-wheel gate, app-independent product-maturity sentinel, and authorized integrated suite pass without collecting legacy surrogate applications
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/033-application-session-state-ownership.md
Reason: ADR-033 defines the remaining destination ownership and exact target recovery contract.
Full plan: Docs/superpowers/plans/2026-07-26-task-646-destination-handoffs.md

1. Extend the owner with Study, Artifact, and ACP channels and migrate producers.
2. Settle Study scope and section independently after restore.
3. Resolve Artifact targets through exact get_chatbook lookup with app-thread generation, restart, cancellation, and unmount settlement guards.
4. Complete current-only ACP session target recovery.
5. Remove every raw pending field and close the AST/privacy boundary.
6. Run focused, app-independent product-maturity, installed-wheel, static, and authorized integrated gates before reconciling TASK-643 through TASK-646 together.
7. Exercise application behavior only through the normal production TldwCli and actual destination screens; use direct tests only for app-independent functions and exclude legacy surrogate applications.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the ADR-033 destination handoff boundary for Study, Artifacts, and ACP while preserving revisioned memory-only single-slot semantics. Study scope/section settlement is independent; Artifacts resolves only the exact canonical local:chatbook record with guarded retry/terminal outcomes; ACP reconstructs and matches only the current canonical local:acp_session record, keeps the exact row selected, exposes its real detail pane, and applies bounded recovery for malformed/stale/unsupported targets. Removed the remaining raw pending fields and dead Notes slot and added AST/privacy ownership guards. Restored the required BaseAppScreen on_mount lifecycle in ACPScreen during self-review.

ADR required: yes
ADR path: backlog/decisions/033-application-session-state-ownership.md
Reason: ADR-033 governs destination ownership, exact-target recovery, settlement, and privacy.

Verification: exact-target/replacement/privacy/full-production-app/static coverage, the app-independent product-maturity sentinel, and the installed-wheel/licence gate all passed in the authorized integrated suite: 473 passed with one pre-existing requests dependency warning in 259.60s. Scoped compileall, Ruff, format, and git diff checks passed. No legacy surrogate application was collected.
<!-- SECTION:NOTES:END -->
