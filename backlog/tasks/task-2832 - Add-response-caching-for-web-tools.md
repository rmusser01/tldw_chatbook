---
id: TASK-2832
title: Add response caching for web tools
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 06:05'
updated_date: '2026-08-08 21:56'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Identical searches/fetches in a session waste API quota and latency. The classic ToolExecutor has ToolResultCache (LRU+TTL+disk) but the agent-runtime/hub path has none. Add hub-side caching for web_search/web_fetch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cache keyed by normalized args with TTL + size bounds,Rate limits still apply on misses; domain-only logging preserved,Tests for hit/miss/expiry
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Spec + adversarial review (Docs/superpowers/specs/2026-08-08-web-search-cache-design.md)
2. Add _search_cache (TTL 900s, 128 entries, earliest-expiry eviction) to web_tool_impls, gated at the single success-blocks point
3. Tests: hit/normalization/expiry/error-shapes-not-cached/eviction/log-privacy/deep-search isolation + the missing _fetch_cache TTL-expiry test
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scope honesty (spec, "What already exists"): web_fetch's cache half shipped in v1 — this task added its one MISSING test (TTL expiry) and implemented the real gap, web_search. Cache lives in the TOOL function only (web_deep_search's phase-1 keeps hitting the live pipeline — pinned by a new real-generate_and_search isolation test); key = post-coercion (engine, whitespace-collapsed casefolded query, count), first-populator's-raw-text-wins recorded; ONLY the genuine success-blocks output is cached, stored at that single code point — the spec review enumerated web_search's five other return shapes (exception, non-dict, error envelope, malformed list, confirmed-empty) and all are deliberately uncached (transient failures and parser-zero histories must not pin for the TTL). AC's "rate limits still apply on misses" recorded as vacuous (no search-side limiter exists); "domain-only logging" read as engine-only, scoped to the wrapper (Bing/Google backend query-logging is pre-existing and out of scope), pinned by a log-privacy test. New hazard found live: the module-level cache POLLUTES any test file invoking web tools without reset — 5 pre-existing provider tests failed on landing; test_local_tool_provider.py gained an autouse _reset_state_for_tests fixture. Mutation checks: cache-store bypass and normalization drop both red→green. Files: web_tool_impls.py, local_tool_provider.py (description), test_web_tool_impls.py (+11 tests), test_local_tool_provider.py (autouse reset).
<!-- SECTION:NOTES:END -->
