---
id: TASK-34932
title: Move-the-Settings-agents-panel-list-read-off-the-event-loop
status: Done
assignee: []
created_date: 2026-10-13 14:54
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: medium
updated_date: 2026-10-13 18:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
settings_agents_panel._reload_list (awaited from on_mount and every save/delete) calls self._runs_db.list_agent_definitions() — a sync file-backed SQLite read inside an async coroutine, and coroutine workers still run on the event loop — then appends rows one await at a time (settings_agents_panel.py:339). Users feel Settings→Agents paint lag at mount and click lag after each save, scaling with definition count.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] list_agent_definitions runs in a thread (to_thread or thread worker)
- [x] Rows appended in one batch instead of one await per row
- [x] Settings→Agents paint no longer blocks the loop on the DB read (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _reload_list + list_agent_definitions\n2. Thread the DB read and batch the row appends\n3. Test + targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_reload_list` (settings_agents_panel.py) now reads definitions via `await asyncio.to_thread(self._runs_db.list_agent_definitions)` — AgentRunsDB uses thread-local connections (the panel's own orphan-reconcile worker already threads DB calls), so the coroutine-worker-on-loop SQLite read is gone — and appends rows via ONE `ListView.extend` batch instead of one awaited `append` per row.
- Tests: new `Tests/UI/test_settings_agents_offloop_read.py` (definition read on a non-loop thread; single batched extend; rows preserved). 1/1 pass. Regression: `Tests/UI/test_settings_agents_category.py` at its exact pre-existing baseline (13 failed / 20 passed with the change stashed).
- ADR required: no — threading + batching inside one panel method.
