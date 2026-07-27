---
id: TASK-887
title: Defer the HuggingFace browse until the download-models view is opened
status: To Do
assignee: []
created_date: '2026-07-27 13:31'
labels:
  - bug
  - performance
  - llm-management
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every visit to the Lab Models screen fires an unrequested live request to huggingface.co. ModelSearchWidget.on_mount schedules call_after_refresh(_initial_browse), which calls perform_search() and reaches HuggingFaceAPI().search_models(...). That widget lives inside llm-view-download-models, which LLMManagementWindow.compose() builds eagerly, so the call happens even for users who never open Download Models. It also contributes to the screen's 488-787ms mount cost. Evidence it is a real problem: two test files in PR #966 had to stub HuggingFaceAPI.search_models to get deterministic runs. The fix is the same principle as the Lab frame's deferred body mount, one level down.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No network request is made on mounting the Models screen,The browse runs when the download-models view is actually activated,The view shows an honest state before its first browse rather than an unexplained empty list,Tests covering Models no longer need to stub HuggingFaceAPI.search_models to be deterministic
<!-- AC:END -->
