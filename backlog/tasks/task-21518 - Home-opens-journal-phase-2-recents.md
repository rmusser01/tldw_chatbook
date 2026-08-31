---
id: TASK-21518
title: Home opens journal (phase 2 recents)
status: To Do
assignee: []
created_date: '2026-08-31 02:43'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persistent opens journal (IDs+timestamps only, model_catalog_cache.json pattern) so read-only sessions count as recent work without edits; feeds recents ranking and task-18921 usage-ranked suggestions (spec 2026-08-29 Non-goals)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening an item bumps its recency even without edits,Journal storage contract documented,Usage-ranked suggestion feasibility assessed against task-18921
<!-- AC:END -->
