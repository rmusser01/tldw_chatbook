---
id: TASK-508
title: Fix recursive_scrape browser-tab leak on page.goto block
status: To Do
assignee: []
created_date: '2026-07-23 12:00'
labels: [web, performance]
dependencies: [task-328]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`recursive_scrape` inner `page.goto` block skips `page.close()` when a link is blocked, causing browser-tab leaks per blocked link. Wrap the block in `try/finally` to ensure cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `recursive_scrape` always closes browser pages in the inner loop, even when goto or extraction fails
- [ ] Browser-tab leaks are eliminated for blocked-link scenarios
<!-- AC:END -->
