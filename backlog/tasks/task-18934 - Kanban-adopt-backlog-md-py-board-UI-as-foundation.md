---
id: TASK-18934
title: 'Kanban: adopt the backlog-md-py board UI as the foundation'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - kanban
  - ui
  - agents
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A kanban surface exists in the app but is not directly accessible and its UI is believed broken or poor (2026-08-19 hermes-release review follow-up; hermes's Kanban is its founding desktop plugin and a proven multi-agent work-board pattern). Rather than patching the current surface, migrate and improve the kanban Textual UI/app from backlog-md-py as the foundation and rebuild chatbook's board on top of it: keyboard-first column/card navigation, card detail view, and a real entry point from the app shell (nav/palette). Foundation work, explicitly not immediate — coordinate with the surfaces that would drive cards (Scheduling module jobs, agent-run wake completions) so the board has honest data sources. Step zero is an audit of what kanban data/surface actually exists today, since the current state is secondhand knowledge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An audit of the existing kanban surface and its data (what exists, what is broken, whether anything is persisted) is recorded in the task before any code lands
- [ ] #2 The backlog-md-py kanban TUI is evaluated and ported/adapted as a self-contained screen or widget: keyboard-first navigation (columns ←/→, cards ↑/↓, Enter opens card detail) with no mouse dependency
- [ ] #3 Existing kanban data (if the audit finds any usable) maps or migrates to the new surface honestly — no silent loss, and dead/unusable legacy data is explicitly reported rather than carried
- [ ] #4 The board is reachable from the app shell (nav bar and command palette) under normal chrome rules; no dead or hidden entry points
- [ ] #5 Keybindings follow ADR-031 (htop-style single-letter screen actions; no terminal-convention keys; footer hints only advertise implemented actions)
- [ ] #6 Tests cover navigation, data mapping, and shell accessibility; a user guide page documents the board
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/075-kanban-board-foundation.md (to be drafted before implementation).
Reason: long-lived application-structure decision — replaces an existing (broken) surface with a ported architecture and defines the board's data ownership (local store vs Scheduling/agent-run sources). Not immediate: sequence after current Console/fleet work settles.

1. Audit the existing kanban surface and data; record findings in this task
2. Draft ADR-075: port vs rewrite boundary, data ownership, integration seams (Scheduling, agent runs)
3. Port/adapt the backlog-md-py board UI as a self-contained Textual surface
4. Shell entry points + keyboard conventions per ADR-031
5. Tests + user guide page
<!-- SECTION:PLAN:END -->
