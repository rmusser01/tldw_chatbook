---
id: TASK-566
title: Cancel settings-rag-index-status workers on category nav
status: To Do
assignee: []
created_date: '2026-07-25 07:57'
labels:
  - settings
  - rag
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The exclusive worker group settings-rag-index-status (SP3-era) lets a stale off-thread status fetch land its callback after the user navigates away from Library/RAG — including, post-541, a re-index confirm modal appearing over an unrelated category. 541 reviews rated it pre-existing/non-blocking; wants a cancel-group-on-nav sweep in _select_category.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leaving the Library/RAG category cancels in-flight index-status workers
- [ ] #2 No modal or status write can land after nav-away (regression test)
- [ ] #3 Re-entry still triggers a fresh fetch
<!-- AC:END -->
