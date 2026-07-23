---
id: TASK-329
title: Add response-size caps and timeouts to remaining fetchers
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
labels: [security, web]
dependencies: [task-328]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Several fetchers read full response bodies with no size cap and, in places, no timeout: `get_page_title` reads `response.text` uncapped (`Web_Scraping/Article_Extractor_Lib.py:295`); the sitemap/crawl `requests.get` calls have no timeout and no cap (`Article_Extractor_Lib.py:889,937`); subscription scrapers read full bodies without a byte cap. A hostile or slow endpoint can hang the app or exhaust memory. `Local_Ingestion/web_article_ingestion.py:68-90` is the correct template (streaming `_MAX_BYTES` cap + content-type allowlist + 30s timeout) and should be mirrored.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All user/model-reachable fetchers enforce a byte cap and a request timeout
- [ ] #2 `get_page_title` and the sitemap/crawl `requests.get` calls specifically gain caps and timeouts
- [ ] #3 Oversized or slow responses fail cleanly with a clear error rather than hanging or OOMing
<!-- AC:END -->
