---
id: TASK-25913
title: Clamp per-tool execution timeout to the run's remaining wall budget
status: To Do
assignee: []
created_date: '2026-08-31 15:10'
updated_date: '2026-08-31 15:11'
labels:
  - agents
  - defect
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A single tool call can run past the run's wall-clock budget. Verified on origin/dev: max_wall_seconds is checked only at the top of the loop (Agents/agent_runtime.py:1175), while _call_with_timeout(fn, seconds, ...) at Agents/agent_service.py:1522 takes an absolute bound with no reference to time already spent. The engine default is 300s (Agents/agent_models.py:400) but Console raises it to 3600s (Chat/console_agent_bridge.py:408), so a hung tool can hold a run roughly an hour past a budget the user set. Found while verifying an area agent's claim during the 2026-08-31 parity pass; this is a defect, not a parity gap - hermes is not the reason to fix it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A tool call's effective timeout is the lesser of the configured per-tool timeout and the run's remaining wall budget
- [ ] #2 A call cut short by the remaining-budget clamp is reported distinctly from one that hit the per-tool ceiling, so the cause is legible
- [ ] #3 Human-approval waits continue to pause the deadline as they do today (ADR-067 refcounted marks) and are not counted against the clamp
- [ ] #4 A run with no wall budget configured behaves exactly as today
- [ ] #5 A test asserts a long tool call cannot push a run past max_wall_seconds
<!-- AC:END -->
