---
id: TASK-328
title: Add SSRF protection to all outbound URL fetching
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
labels: [security, web]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is no SSRF protection anywhere in the codebase. `validate_url` (`Utils/input_validation.py:105-162`) checks only scheme/syntax and intentionally permits `localhost`, `127.0.0.1`, `169.254.169.254`, and internal hostnames (its own docstring says so); no private/loopback/link-local guard exists project-wide. Every fetcher follows redirects (`follow_redirects` / `allow_redirects=True`) with no post-redirect host re-validation, so a validated public URL that returns `302 -> http://169.254.169.254/` is followed. `Scraper.page.goto()` (`Web_Scraping/Article_Scraper/scraper.py:158`) has no URL validation at all and is subclassed by the user-configured `ConfluenceScraper`. This is primarily a user-driven-ingestion risk (no tool exposes a raw fetch to the model), but the gap is broad and affects article extraction, web ingestion, subscription scrapers, and Confluence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A shared guard rejects private/loopback/link-local/reserved/multicast hosts (by resolved IP) and non-http(s) schemes including `file:`
- [ ] #2 The guard is enforced pre-request and re-checked on every redirect hop
- [ ] #3 The guard is wired into `validate_url` (or a new `validate_public_url`) and into `Scraper._fetch_html` / `page.goto` plus the other fetch sites (Article_Extractor_Lib, web_article_ingestion, subscription scrapers, confluence)
- [ ] #4 Tests cover metadata-IP, loopback, RFC1918, `file://`, and redirect-to-internal cases
<!-- AC:END -->
