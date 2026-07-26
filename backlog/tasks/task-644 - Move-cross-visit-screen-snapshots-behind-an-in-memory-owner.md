---
id: TASK-644
title: Move cross-visit screen snapshots behind an in-memory owner
status: To Do
assignee:
  - '@codex'
created_date: '2026-07-26 13:35'
labels:
  - architecture
  - state
  - ui
dependencies:
  - TASK-643
references:
  - backlog/decisions/026-application-session-state-ownership.md
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
- [ ] #1 ScreenStateStore owns canonical-tab snapshot envelopes in memory, rejects off-owner mutation, treats restored inputs as read-only, and never inserts policy metadata into domain snapshot dictionaries
- [ ] #2 Runtime-source or active-server incompatibility discards stale snapshots while compatible snapshots restore without exposing the backing mapping
- [ ] #3 TldwCli navigation no longer creates or reads _screen_states and recent-snapshot consumers use the truthful store API
- [ ] #4 Fresh screen construction, canonical alias-configured startup, alias sharing by resolved canonical tab, pending-work vetoes, corrupt snapshot recovery, and explicit Library, Settings, and Watchlists navigation-context precedence are preserved
- [ ] #5 Focused mounted, nested Settings-draft, large Console snapshot, off-owner mutation, payload-redaction sentinel, scoped static, and ownership-guard checks pass
<!-- AC:END -->
