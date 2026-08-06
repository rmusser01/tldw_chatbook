---
id: TASK-1357
title: Add recursive web crawling as a tool
status: To Do
assignee: []
created_date: '2026-08-05 06:04'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Article_Extractor_Lib already implements sitemap and recursive crawlers (scrape_from_sitemap, sync_recursive_scrape, scrape_by_url_level). Wrap them in a budgeted web_crawl tool behind the egress guard — page list returned to the model, no permanent ingestion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 web_crawl with max_pages/max_depth budgets + egress guard,Per-domain rate limiting honored,Ask default; results ephemeral; tests with local fixture site
<!-- AC:END -->
