---
id: TASK-652
title: Remove duplicate Media root state and stop mutation bubbling
status: To Do
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 00:16'
labels:
  - architecture
  - state
  - media
  - reliability
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
Make MediaWindow the sole Media view and selection owner and ensure one production Media event performs one scoped mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Media destination root reactives, duplicate search generation/timers, and all of their writers, watchers, and legacy emitters are removed; legacy Chat sidebar pagination and selection fields remain assigned to TASK-650.
- [ ] #2 The Media destination stops mutation events before awaiting work and the duplicate app-level mutation handler registration is removed.
- [ ] #3 One real metadata event performs exactly one scoped mutation and refresh, while older item-detail and search completions cannot overwrite newer destination state.
- [ ] #4 MediaScreen snapshots and restores only the actual MediaWindow owner.
- [ ] #5 Normal production TldwCli Media checks plus focused ownership, static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->
