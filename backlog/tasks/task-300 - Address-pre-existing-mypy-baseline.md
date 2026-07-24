---
id: TASK-300
title: Address pre-existing mypy baseline
status: To Do
assignee: []
created_date: '2026-07-19 16:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The codebase currently reports ~4,000 mypy errors when run project-wide. Reduce the baseline incrementally so CI can enforce type cleanliness across all tldw_chatbook modules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Project-wide mypy run exits cleanly,CI enforces mypy for all source files,Existing errors are fixed or have documented ignores
<!-- AC:END -->
