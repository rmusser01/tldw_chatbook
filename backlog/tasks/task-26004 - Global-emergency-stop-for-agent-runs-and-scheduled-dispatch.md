---
id: TASK-26004
title: Global emergency stop for agent runs and scheduled dispatch
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:43'
updated_date: '2026-09-01 21:04'
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
- [x] #1 A single user-visible control stops new agent runs and new scheduled dispatches from starting
- [x] #2 In-flight agent runs and already-dispatched scheduled tasks are NOT killed; they run to completion
- [x] #3 The stop survives an application restart
- [x] #4 The state is fail-safe: if reading it errors, the system treats it as stopped rather than proceeding
- [x] #5 While stopped, an attempted send or dispatch reports plainly that the stop is active and how to clear it
- [x] #6 Clearing the stop resumes normal operation without requiring a restart
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. emergency_stop.py sentinel: durable JSON, fail-SAFE read (error=>stopped), set/clear/is_stopped/state+visible_copy (TDD)\n2. Scheduler reader: _dispatch_due checks estop BEFORE pop_due (holds without consuming; resumes on clear)\n3. Agent reader: send_refusal_copy returns visible_copy when stopped (in-flight untouched)\n4. Control: /emergency-stop [clear|<reason>] command + grammar/suggestion/screen wiring + order-pin updates; guide
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One sentinel, several readers (the honest shape the task asked for). tldw_chatbook/emergency_stop.py: a durable JSON switch (atomic write) with read_emergency_stop that FAILS SAFE — a missing file is not-stopped, but a corrupt/unreadable state returns active=True (AC#4: halt on doubt, the safe direction for an emergency stop). is_emergency_stopped (hot boolean) + emergency_stop_state (full state + visible_copy naming the stop and the clear step, AC#5). Readers: (1) SchedulerLoop._dispatch_due checks estop BEFORE queue.pop_due, so a stop HOLDS new dispatches without draining/consuming them — held tasks stay queued and fire when cleared (AC#1/#2/#6), reader is getattr-defensive for hand-wired tests; (2) console send_refusal_copy returns the estop visible_copy first, refusing every new send while in-flight runs are untouched (AC#1/#2/#5). Control: /emergency-stop [clear|<reason>] Console command (grammar + suggestion + screen handler; order pins updated like /redirect) — clear/off/resume lifts it, anything else sets it with an optional reason (AC#1 single control, AC#6 clear-without-restart). Durable across restart (AC#3, JSON file in user data dir). 8 estop tests (sentinel + scheduler-hold) + command grammar/suggestion pins; scheduler suite 404, console command/queue suites 62 green. Scope note: the control is a Console command (the primary agent surface); a Home/settings button could be added later, but the sentinel is control-surface-agnostic so any future affordance just calls set/clear.
<!-- SECTION:NOTES:END -->
