---
id: TASK-644
title: Move cross-visit screen snapshots behind an in-memory owner
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 13:35'
updated_date: '2026-07-26 21:35'
labels:
  - architecture
  - state
  - ui
dependencies:
  - TASK-643
references:
  - backlog/decisions/033-application-session-state-ownership.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-26-application-session-state-ownership-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace TldwCli's raw screen-state dictionary with a process-memory owner that preserves fresh-screen navigation, runtime-scope compatibility, and explicit deep-link precedence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ScreenStateStore owns canonical-tab snapshot envelopes in memory, rejects off-owner mutation, treats restored inputs as read-only, and never inserts policy metadata into domain snapshot dictionaries
- [x] #2 Runtime-source or active-server incompatibility discards stale snapshots while compatible snapshots restore without exposing the backing mapping
- [x] #3 TldwCli navigation no longer creates or reads _screen_states and recent-snapshot consumers use the truthful store API
- [x] #4 Fresh screen construction, canonical alias-configured startup, alias sharing by resolved canonical tab, pending-work vetoes, corrupt snapshot recovery, and explicit Library, Settings, and Watchlists navigation-context precedence are preserved
- [x] #5 Focused mounted, nested Settings-draft, large Console snapshot, off-owner mutation, payload-redaction sentinel, scoped static, and ownership-guard checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/033-application-session-state-ownership.md
Reason: ADR-033 defines application-lifetime snapshot ownership, canonical keying, runtime invalidation, and memory-only persistence.
Full plan: Docs/superpowers/plans/2026-07-26-task-644-screen-state-store.md

1. Add the owner-thread-affine memory-only ScreenStateStore.
2. Route navigation through canonical keys while preserving flush/save/restore/context/switch order.
3. Remove duplicate startup current_tab publication.
4. Migrate recent-work consumers without crossing worker-thread boundaries.
5. Prove nested Settings copying, large Console shallow storage, and final ownership guards, then keep TASK-644 In Progress until the shared TASK-646 release gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-033 memory-only ScreenStateStore and migrated TldwCli navigation plus recent-work consumers to its canonical, runtime-scoped API. Navigation preserves fresh production screen construction, canonical alias startup and sharing, pending-work veto/failure behavior, corrupt-restore discard, nested Settings isolation, shallow large-payload storage, and restore-before-explicit-context ordering. Added Tests/UI/test_screen_state_full_app.py using only the normal production TldwCli and actual destination screens; app-independent store and ownership behavior remains directly tested.

ADR required: yes
ADR path: backlog/decisions/033-application-session-state-ownership.md
Reason: ADR-033 governs memory-only snapshot ownership, canonical keys, runtime invalidation, and copy boundaries.

Verification: the focused ScreenStateStore/full-production-app/ownership gate passed after adding complete flush-save-construct-restore-context-switch ordering coverage; the authorized integrated suite passed 473 tests with one pre-existing requests dependency warning in 259.60s. Scoped compileall, Ruff, format, boundary-pattern, and git diff checks passed. The plan was corrected to exclude, never run, and never cite retired surrogate application suites.
<!-- SECTION:NOTES:END -->
