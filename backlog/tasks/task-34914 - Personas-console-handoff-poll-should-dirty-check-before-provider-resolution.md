---
id: TASK-34914
title: Personas-console-handoff-poll-should-dirty-check-before-provider-resolution
status: To Do
assignee: []
created_date: '2026-10-13 14:44'
labels:
  - performance
  - ui-perf
  - wave-1-timers
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Personas 0.25s _poll_console_handoff_readiness timer calls _provider_send_block_reason() (full console_handoff_readiness provider resolution) every tick before comparing the result to the cached reason (personas_screen.py:7296-7313). Same work-before-gate shape as the Settings readiness poll: 4Hz provider resolution while Personas is mounted even when inputs are unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 console_handoff_readiness runs only when cheap inputs (header inputs and provider selection) change,Idle ticks skip provider resolution (test asserts resolution call count),Handoff gating behavior unchanged
<!-- AC:END -->
