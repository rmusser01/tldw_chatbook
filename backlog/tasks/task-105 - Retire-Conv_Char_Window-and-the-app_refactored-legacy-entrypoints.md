---
id: TASK-105
title: Retire Conv_Char_Window and the app_refactored legacy entrypoints
status: To Do
assignee: []
created_date: '2026-06-11 23:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CCPWindow (Conv_Char_Window.py) is reachable only from app_refactored_v2.py's lazy fallback, which has no in-package consumers; app_refactored.py and tldw_chatbook/navigation/screen_registry.py are similarly dead. Retire them together after confirming nothing external depends on the alternate entrypoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No dead alternate entrypoints remain,CCP_Modules TYPE_CHECKING hints updated
<!-- AC:END -->
