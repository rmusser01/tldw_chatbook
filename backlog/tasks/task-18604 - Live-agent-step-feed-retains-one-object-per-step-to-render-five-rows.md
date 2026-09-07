---
id: TASK-18604
title: Live agent step feed retains one object per step to render five rows
status: Done
assignee: []
created_date: '2026-08-18 21:20'
labels:
  - agents
  - console
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConsoleAgentBridge.run_reply` kept each run's live step feed as a plain list,
appended once per step and read in exactly two ways: `len(...)` for the rail's
step counter and `[-5:]` for its recent-step rows. Nothing ever read the middle,
yet the list retained one `AgentLiveStep` per step for the life of the run.

At the run budget's raised step ceiling (TASK-18600) that is up to 25,000
retained objects per run to serve a five-row display, multiplied by each live
sub-agent's own feed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The live feed retains a bounded number of steps regardless of run length.
- [x] #2 The rail's step counter still reports the true total, not the retained count.
- [x] #3 The rail's recent-step rows are unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the list with a small `_LiveStepFeed` holding a monotonic `count` and a
`deque(maxlen=8)` tail.

Keeping the count and the tail in one object, rather than swapping in a bare
`deque(maxlen=...)`, is the point: a bare deque silently redefines `len()` as
"how many we kept", which is exactly the wrong number for a step counter that
the rail displays. The type makes that impossible to get wrong.

Files: `Chat/console_agent_bridge.py`.
<!-- SECTION:NOTES:END -->
