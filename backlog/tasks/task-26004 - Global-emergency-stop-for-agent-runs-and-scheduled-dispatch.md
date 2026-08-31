---
id: TASK-26004
title: Global emergency stop for agent runs and scheduled dispatch
status: To Do
assignee: []
created_date: '2026-08-31 15:43'
labels:
  - agents
  - scheduling
  - ops
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is no way to stop everything at once. Verified on origin/dev: agent cancellation is per-session (Chat/console_chat_controller.py:13048) and the only broad path is shutdown() at :13129, which tears down at exit; on the scheduling side a named grep for estop and emergency stop across tldw_chatbook/Scheduling returns zero, so tasks can only be disabled one at a time. Hermes uses a single sentinel checked by the cron scheduler, the kanban dispatcher and the gateway, stopping NEW work while leaving in-flight runs untouched, fail-safe if the check itself errors. Filed once covering both consumers because that is the honest shape - one switch, several readers - rather than two unrelated features.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single user-visible control stops new agent runs and new scheduled dispatches from starting
- [ ] #2 In-flight agent runs and already-dispatched scheduled tasks are NOT killed; they run to completion
- [ ] #3 The stop survives an application restart
- [ ] #4 The state is fail-safe: if reading it errors, the system treats it as stopped rather than proceeding
- [ ] #5 While stopped, an attempted send or dispatch reports plainly that the stop is active and how to clear it
- [ ] #6 Clearing the stop resumes normal operation without requiring a restart
<!-- AC:END -->
