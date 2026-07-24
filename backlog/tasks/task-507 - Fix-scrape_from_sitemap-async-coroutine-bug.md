---
id: TASK-507
title: Fix scrape_from_sitemap/scrape_by_url_level async coroutine bug
status: To Do
assignee: []
created_date: '2026-07-23 12:00'
labels: [web, bug]
dependencies: [task-328]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`scrape_from_sitemap` and `scrape_by_url_level` call `async scrape_article` without `await`, returning un-awaited coroutines. This means discovered-URL scraping is inert; the SSRF guard and other egress validations from TASK-328 never run for these URLs. Make `scrape_from_sitemap` and related callers async (ripples up the call chain) OR gather the coroutines and await them concurrently. Pre-existing bug that blocks TASK-328's trust-threading from taking effect on discovery-sourced URLs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `scrape_from_sitemap` and `scrape_by_url_level` properly await async scrape_article calls or gather them for concurrent execution
- [ ] Discovered URLs are now subject to SSRF guards and other egress validations from TASK-328
- [ ] Tests verify that discovered-URL scraping actually executes (coroutines are not left unawaited)
<!-- AC:END -->
