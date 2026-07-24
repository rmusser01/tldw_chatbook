---
id: TASK-505
title: Confluence replace sync requests calls inside async methods
status: To Do
assignee: []
created_date: '2026-07-23 12:00'
labels: [subscriptions, performance]
dependencies: [task-328]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConfluenceAuth.make_request` and `ConfluenceScraper._extract_page_id_from_url` make synchronous `requests` library calls from inside `async def` methods, blocking the event loop for up to the full request timeout (30s after task-328's fix; previously unbounded). This is a performance hazard that can starve concurrent operations. Port these to httpx async or run via a thread executor to avoid blocking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] No synchronous HTTP call remains on the event loop in `Web_Scraping/Confluence/` (ConfluenceAuth and ConfluenceScraper)
- [ ] Async methods use either httpx async or run_in_executor for blocking sync calls
- [ ] Tests verify no blocking calls are made from async contexts
<!-- AC:END -->
