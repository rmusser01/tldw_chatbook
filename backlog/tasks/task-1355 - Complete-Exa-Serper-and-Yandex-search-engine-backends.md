---
id: TASK-1355
title: 'Complete Exa, Serper, and Yandex search engine backends'
status: To Do
assignee: []
created_date: '2026-08-05 06:03'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_web_serper and search_web_yandex are empty stubs in Web_Scraping/WebSearch_APIs.py and Exa is absent entirely, so the search_engine enum offers dead options. Complete the three backends so engine choice is real.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 serper+yandex implemented and wired into perform_websearch,Exa added with API call + result parsing + [SearchEngines] key,Unit tests with mocked responses + optional live tests
<!-- AC:END -->
