---
id: TASK-1360
title: Add response caching for web tools
status: To Do
assignee: []
created_date: '2026-08-05 06:05'
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
- [ ] #1 Cache keyed by normalized args with TTL + size bounds,Rate limits still apply on misses; domain-only logging preserved,Tests for hit/miss/expiry
<!-- AC:END -->
