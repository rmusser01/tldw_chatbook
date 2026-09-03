---
id: TASK-31204
title: Export destination path normalization bypasses shared path validation
status: To Do
assignee: []
created_date: '2026-09-03 19:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qodo finding on PR #2344 (library_export_controller.py ~:1275, relocated verbatim from LibraryScreen._apply_library_export_destination): the normalized destination path is not run through path_validation.py's shared checks. Pre-existing behavior surfaced by the extraction; fix in the isolated controller, not in the move PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Destination path flows through the shared path-validation seam
- [ ] #2 Covering test exercises a hostile path input
<!-- AC:END -->
