---
id: TASK-33407
title: Measure and, if it pays, coalesce the SSH session's admitted marker and result
status: To Do
assignee: []
created_date: '2026-09-28 20:30'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The live UAT for PR #2879 saw a fast operation's result arrive ~8 ms after its admitted marker while the host work took ~4 ms, suggesting the second small write waits for the first's acknowledgement. Sending both in one write could save about one round trip per warm call. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The effect is measured on the live host with interleaved before/after runs and same-window ping
- [ ] #2 If warm-call latency improves measurably, a fast operation's marker, result and status leave the host in one write while a slow operation's marker is still sent promptly
- [ ] #3 If it does not improve, no coalescing code ships and the measurement is recorded
<!-- AC:END -->
