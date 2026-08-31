---
id: TASK-26001
title: 'Agent loop: graceful wrap-up before budget exhaustion'
status: To Do
assignee: []
created_date: '2026-08-31 15:43'
labels:
  - agents
  - reliability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A run that exhausts its budget dies with a bare error and no answer. Verified on origin/dev: Agents/agent_runtime.py:1169-1179 checks four budgets at the loop top and on breach adds STEP_ERROR "step budget exhausted" then returns RUN_STUCK - the user gets nothing usable from work already done. Hermes appends a one-time wrap-up notice to the newest tool message at roughly 80 percent of the wall budget (cache-safe, no synthetic user turn) and on exhaustion makes one tools-stripped call so the user gets a summary instead of a dead run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a run passes a configurable fraction of its wall or step budget, the model is told once, without inserting a synthetic user turn
- [ ] #2 The notice is attached so it does not break the prompt-cache stable prefix - verified and recorded in the notes
- [ ] #3 On budget exhaustion the loop makes one final model call with tools removed, and its output is presented as the run result
- [ ] #4 The final wrap-up call is itself bounded and cannot loop or spawn tools
- [ ] #5 If the wrap-up call fails, the run still terminates honestly with the existing exhaustion message rather than hanging
- [ ] #6 Budget-exhausted runs remain distinguishable from successful completion in the run record
<!-- AC:END -->
