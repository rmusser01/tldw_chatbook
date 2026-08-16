---
id: TASK-16326
title: Unify academic search into the deep-search pipeline
status: To Do
assignee:
  - '@robert'
created_date: '2026-08-15 05:16'
labels:
  - research
  - web-tools
dependencies:
  - TASK-16322
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
LocalResearchSearchService implements arXiv and Semantic Scholar inline with urllib, hardcoded 30s timeouts, no API-key support, no retry or backoff, and no connection to the deep-search evidence pool. tldw_server treats academic as just another lane feeding one evidence graph with DOI dedup. Migrate the runners to httpx with retries and let academic results feed the same evidence pool as web results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 arXiv and Semantic Scholar runners use httpx with configurable timeouts and retry with backoff,API keys are read from config when available for providers that support them,Academic results feed the same evidence pool as web results with DOI-level dedup,Provider endpoints and engine lists are config or constants driven rather than inline literals,Tests with mocked HTTP cover retry and dedup behavior
<!-- AC:END -->
