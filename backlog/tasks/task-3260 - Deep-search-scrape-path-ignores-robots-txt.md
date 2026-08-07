---
id: TASK-3260
title: Deep-search scrape path ignores robots.txt
status: To Do
assignee: []
created_date: '2026-08-07 21:15'
labels:
  - web-tools
dependencies:
  - TASK-2833
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-2833 added robots.txt enforcement for web_fetch and web_crawl (per-domain fetch+cache, Disallow honored, `[webfetch] respect_robots_txt` toggle), but its design doc recorded a non-goal: web_deep_search's scrape path (`Web_Scraping.WebSearch_APIs`'s per-result relevance/summarization scraping, invoked from `analyze_and_aggregate`) was left out of scope.

This leaves an inconsistency: a path that web_fetch/web_crawl now refuse to fetch (per that host's robots.txt) can still be scraped by web_deep_search, because its scrape path never consults robots.txt at all. The codebase already has precedent for closing exactly this kind of gap — `is_public_http_url`'s SSRF check was bolted onto `scrape_article` after the same shape of inconsistency was noticed between the SSRF-guarded tools and the deep-search scrape path — so the fix shape here is the same: thread a robots.txt consult into the scrape path rather than carving out a permanent architectural exemption.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 web_deep_search's scrape path (the per-result fetch inside `analyze_and_aggregate`/its scraping helper) is refused for a URL disallowed by that host's robots.txt, under the same `[webfetch] respect_robots_txt` toggle task-2833 introduced (default on, fail-open on an unreachable/unparsable robots.txt)
- [ ] #2 A robots-disallowed result is skipped (not fatal to the overall deep-search run), mirroring web_crawl's skip-and-count behavior rather than web_fetch's hard refusal
- [ ] #3 Fixture-based tests (MockTransport, no live network) cover: a disallowed scrape URL is skipped, an allowed one proceeds, and the toggle-off path makes no robots.txt fetch at all
<!-- AC:END -->
