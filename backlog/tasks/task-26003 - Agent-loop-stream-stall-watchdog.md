---
id: TASK-26003
title: 'Agent loop: stream stall watchdog'
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
A provider emitting keep-alive bytes without content holds a run until the wall budget expires. Verified on origin/dev: the only bound on a silent stream is the httpx read timeout at Chat/console_provider_gateway.py:150, and a named grep for stall, idle_timeout, no_activity, watchdog and first_token_timeout across Chat/console_provider_gateway.py and Chat/console_agent_bridge.py returns only install* false positives. Because bytes keep arriving, the read timeout never fires. Hermes tracks last-content time and trips a cross-turn circuit breaker after repeated stale kills.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stream producing no new content for a configurable interval is terminated even while transport-level bytes continue to arrive
- [ ] #2 The distinction between no bytes and no content is explicit: keep-alive and heartbeat frames do not reset the content clock
- [ ] #3 A stall termination is reported distinctly from a network error and from a user cancel
- [ ] #4 Repeated stalls against the same provider within a session surface a warning rather than silently retrying forever
- [ ] #5 Normal slow generation (long thinking, large responses) does not trip the watchdog - verified with a slow but productive stream
<!-- AC:END -->
