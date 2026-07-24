---
id: TASK-328
title: Add SSRF protection to all outbound URL fetching
status: Done
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
- [x] #1 A shared guard rejects private/loopback/link-local/reserved/multicast hosts (by resolved IP) and non-http(s) schemes including `file:`
- [x] #2 The guard is enforced pre-request and re-checked on every redirect hop
- [x] #3 The guard is wired into `validate_url` (or a new `validate_public_url`) and into `Scraper._fetch_html` / `page.goto` plus the other fetch sites (Article_Extractor_Lib, web_article_ingestion, subscription scrapers, confluence)
- [x] #4 Tests cover metadata-IP, loopback, RFC1918, `file://`, and redirect-to-internal cases
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `tldw_chatbook/Utils/egress.py` — single SSRF policy (allow iff every resolved IP public & not cloud-metadata, OR host in trusted_origins, OR in [web_security] allowed_hosts; metadata blocked even for trusted, only allowlist overrides) with sync+async DNS eval (async never blocks the loop). Guarded fetch helpers for httpx (sync+async), requests (session-auth-safe hops + sink streaming), and aiohttp each doing per-hop redirect re-validation with cross-origin credential stripping. Playwright paths: pre-goto check + post-navigation redirect-chain validation. Wired: Article_Extractor_Lib, Article_Scraper (crawler+Scraper), Confluence, web_article_ingestion, Subscriptions monitors + watchlists + 6 scrapers, media/audio downloaders. Subscriptions/security.py's dead validator now delegates to egress. Fail-closed trust threading (shared functions never auto-trust input URL; only user-boundaries seed trust via origin_set). New [web_security] config (enabled kill-switch + allowed_hosts). Residuals documented: DNS-rebinding TOCTOU, Playwright mid-chain GET (response discarded). Files: Utils/egress.py + config.py + the wired modules. Tests: Tests/Utils/test_egress.py + per-surface wiring tests.
<!-- SECTION:NOTES:END -->
