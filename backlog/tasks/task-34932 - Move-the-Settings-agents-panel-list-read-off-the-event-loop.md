---
id: TASK-34932
title: Move-the-Settings-agents-panel-list-read-off-the-event-loop
status: To Do
assignee: []
created_date: '2026-10-13 14:54'
labels:
  - performance
  - ui-perf
  - wave-4-blocking-rebuilds
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
settings_agents_panel._reload_list (awaited from on_mount and every save/delete) calls self._runs_db.list_agent_definitions() — a sync file-backed SQLite read inside an async coroutine, and coroutine workers still run on the event loop — then appends rows one await at a time (settings_agents_panel.py:339). Users feel Settings→Agents paint lag at mount and click lag after each save, scaling with definition count.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 list_agent_definitions runs in a thread (to_thread or thread worker),Rows appended in one batch instead of one await per row,Settings→Agents paint no longer blocks the loop on the DB read (test)
<!-- AC:END -->
