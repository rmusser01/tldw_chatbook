---
id: TASK-112
title: Debounce Personas library search input
status: To Do
assignee: []
created_date: '2026-06-11 03:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Search re-renders per keystroke; add ~200ms debounce in the search pipeline to cut render churn and FTS query volume.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing quickly triggers at most one render per debounce window,Final query always renders
<!-- AC:END -->
