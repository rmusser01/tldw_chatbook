---
id: TASK-904
title: Remove retired Notes Search Ingest Tools and Evals root state
status: To Do
assignee: []
created_date: '2026-07-26 23:50'
labels:
  - architecture
  - state
  - cleanup
dependencies:
  - TASK-647
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Delete unreachable or no-op root reactives and companion defaults and timers for production destinations that already own their state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes field, sort, preview, and autosave state; Search active-subtab state; Ingest active-view state; Tools active-view state; and Evals sidebar state are removed with every writer, watcher, initializer, timer, and dynamic reference.
- [ ] #2 Library, Search, rebuilt Ingest, MCP, and Evals production destinations remain the only owners of their view state.
- [ ] #3 No compatibility root properties or mirrored state are introduced.
- [ ] #4 The normal production TldwCli can navigate to and exercise every affected registered destination without removed-name access.
- [ ] #5 Focused ownership, static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->
