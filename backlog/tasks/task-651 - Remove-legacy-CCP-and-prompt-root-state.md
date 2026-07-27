---
id: TASK-651
title: Remove legacy CCP and prompt root state
status: To Do
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 00:10'
labels:
  - architecture
  - state
  - personas
  - prompts
dependencies:
  - TASK-647
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove root state and stale callbacks for the retired CCP and prompt editor so Personas and Library remain the only production owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ccp_active_view, CCP provider state, editing and current CCP or conversation identifiers, current prompt state, and their root watchers and handlers are removed.
- [ ] #2 Companion character-image, search-timer, generation, and dead-initializer state is removed.
- [ ] #3 Production import completion refreshes the mounted real owner or defers to a fresh owner load without old widget identifiers or a root cache.
- [ ] #4 Canonical Personas and Library prompt flows pass in the normal production TldwCli.
- [ ] #5 Focused ownership, privacy, static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->
