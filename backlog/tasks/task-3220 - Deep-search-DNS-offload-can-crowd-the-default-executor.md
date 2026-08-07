---
id: TASK-3220
title: Deep-search DNS offload can crowd the default executor
status: To Do
assignee: []
created_date: '2026-08-07 16:30'
labels:
  - web-tools
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-1356's SSRF guard runs DNS resolution via asyncio.to_thread + wait_for (Utils/egress.py is_public_http_url, called from search_result_relevance). When the wait_for timeout fires, the abandoned getaddrinfo thread keeps occupying a default-executor slot until the OS resolver gives up. The relevance loop's own chat_api_call/scrape offloads share that same default executor, so a result set full of slow-DNS hosts can queue paid LLM calls behind abandoned resolvers. Bounded and unlikely, but a direct consequence of the required fix shape — flagged as a deferred minor in Task 3's review and promoted to a follow-up by the final whole-branch review (2026-08-07).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sustained slow-DNS result sets cannot starve the relevance loop's LLM/scrape offloads of executor slots (dedicated bounded executor for DNS, or an equivalent isolation mechanism)
- [ ] #2 A test simulates N consecutive DNS timeouts and shows relevance LLM calls still proceed without waiting on abandoned resolver threads
<!-- AC:END -->
