---
id: TASK-18601
title: >-
  Agent run step log is a single JSON blob column and does not scale to the
  raised step budget
status: To Do
assignee: []
created_date: '2026-08-18 20:30'
labels:
  - agents
  - database
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AgentRunsDB.append_steps` stores a run's entire step log as one JSON string in
the `agent_runs.steps` column: it SELECTs the existing blob, `json.loads` it,
extends the list, and `json.dumps` the whole thing back. `AgentService._persist`
calls it once at end of run, so the write itself is O(n), not O(n^2).

That design was sized for the Console's old 96-step budget. TASK-18600 raised
the shipped step budget to 25000 (owner decision: allow long-running, expensive
sessions). Each step's `result` is capped at 2000 characters by
`agent_runtime.run_agent_loop`, so a worst-case run can now serialize a
tens-of-megabytes JSON blob into one column -- and re-parse all of it every time
the run log is opened, on the UI thread.

Nothing observed failing yet: this is a scaling limit reached by a deliberate
config change, filed at the time the ceiling was raised rather than after a user
hits it. Realistic runs are far smaller (most steps are a few hundred bytes),
which is why TASK-18600 shipped the number as specified instead of lowering it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A run with 25000 recorded steps persists and re-opens without a user-visible stall in the run-log viewer.
- [ ] #2 Reading a run's metadata (status, budget, result) does not require parsing its full step log.
- [ ] #3 The run-log viewer can render a long run without holding every step in memory at once.
- [ ] #4 Existing runs stored in the current blob format remain readable after the change.
<!-- AC:END -->
