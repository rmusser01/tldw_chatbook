---
id: TASK-1361
title: Enforce robots.txt for web tools
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
v1 documents but does not enforce robots.txt. Add per-domain robots fetch+cache and disallow-rule enforcement for tool-initiated fetches/crawls, behind a config toggle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 robots.txt fetched+cached per domain,Disallow honored for web_fetch/web_crawl,[webfetch] respect_robots_txt toggle; fixture-based tests
<!-- AC:END -->
