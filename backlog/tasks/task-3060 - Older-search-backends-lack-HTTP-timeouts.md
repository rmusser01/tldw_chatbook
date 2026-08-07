---
id: TASK-3060
title: Older search backends lack HTTP timeouts
status: To Do
assignee: []
created_date: '2026-08-07 14:10'
labels:
  - web-tools
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The seven older web-search backends (google, bing, brave, duckduckgo, kagi, tavily, searx) plus the baidu stub issue requests.post/requests.get calls with no timeout parameter. An unresponsive provider API hangs perform_websearch (and by extension generate_and_search / the deep-search pipeline) indefinitely, with no way for a caller to bound worst-case latency. Task-1356's phase-1 hardening added timeout=30 to the newer serper/exa/yandex backends (task-1355) but deliberately left these seven+baidu out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every backend HTTP call (google, bing, brave, duckduckgo, kagi, tavily, searx, baidu) carries an explicit timeout
- [ ] #2 A simulated hang/unresponsive provider surfaces as a bounded-time error instead of blocking indefinitely
- [ ] #3 Existing request-shape tests for each backend assert the timeout value
<!-- AC:END -->
