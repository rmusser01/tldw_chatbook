---
id: TASK-3220
title: Deep-search DNS offload can crowd the default executor
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 16:30'
updated_date: '2026-08-07 18:51'
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
- [x] #1 Sustained slow-DNS result sets cannot starve the relevance loop's LLM/scrape offloads of executor slots (dedicated bounded executor for DNS, or an equivalent isolation mechanism)
- [x] #2 A test simulates N consecutive DNS timeouts and shows relevance LLM calls still proceed without waiting on abandoned resolver threads
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a dedicated bounded ThreadPoolExecutor for the SSRF DNS check (module-level, small max_workers, named threads)\n2. Route search_result_relevance's is_public_http_url offload through it via loop.run_in_executor instead of asyncio.to_thread\n3. Test: saturate the DNS executor with blocked resolvers and prove relevance LLM offloads still proceed on the default executor
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: added a module-level, lazily-created, bounded (max_workers=4)
concurrent.futures.ThreadPoolExecutor (`_get_dns_guard_executor`,
thread_name_prefix="deep-search-dns-guard") dedicated solely to the
pre-scrape SSRF DNS guard (is_public_http_url) offload in
search_result_relevance. Routed via
asyncio.get_running_loop().run_in_executor(executor, is_public_http_url,
url) (positional args) instead of asyncio.to_thread, which uses the shared
default executor. wait_for's timeout and the refusal->fallback semantics
are unchanged -- only the executor changed. Double-checked lazy init under
a threading.Lock (same pattern as Chat/console_generate_image.py's
_LLM_CONTEXT_EXECUTOR).

Tests (Tests/Web_Scraping/test_deep_search_pipeline.py):
- test_relevance_guard_runs_on_dedicated_dns_guard_executor: wiring proof --
  captures the worker thread name the guard actually ran on and asserts it
  starts with "deep-search-dns-guard".
- test_dns_guard_executor_saturation_does_not_starve_default_executor_offloads:
  behavioral proof -- fills all 4 dedicated-pool slots with
  threading.Event-blocked fakes (standing in for abandoned resolver
  threads), then shows an asyncio.to_thread offload on the DEFAULT executor
  (the same primitive search_result_relevance uses for chat_api_call)
  still completes in well under the saturation window. No real DNS, no
  real sleeps; unblocking happens deterministically in `finally`.

Files: tldw_chatbook/Web_Scraping/WebSearch_APIs.py,
Tests/Web_Scraping/test_deep_search_pipeline.py.
<!-- SECTION:NOTES:END -->
