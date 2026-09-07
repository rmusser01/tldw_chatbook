---
id: TASK-31205
title: Collections quick-capture fields bypass shared input validation
status: To Do
assignee: []
created_date: '2026-09-03 19:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qodo finding on PR #2344 (library_collections_controller.py ~:871, relocated verbatim from LibraryScreen._submit_library_collection_quick_capture): quick-capture URL/title/tags are not run through input_validation.py's shared checks. Pre-existing behavior surfaced by the extraction; fix in the isolated controller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Quick-capture fields flow through the shared input-validation seam
- [ ] #2 Covering test exercises hostile field input
<!-- AC:END -->
