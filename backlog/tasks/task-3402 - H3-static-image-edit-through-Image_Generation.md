---
id: TASK-3402
title: H3 static image edit through Image_Generation
status: To Do
assignee: []
created_date: '2026-08-09 04:39'
labels:
  - image
  - generation
  - comfyui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Package a sanitized H3 static image-edit workflow inside the existing Image_Generation validation, attachment-storage, and metadata boundary. The sanitized copy removes nodes 154 and 166, and node 165 is the canonical edited-image output. No prompt text, raw export artifact, or source identity is recorded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A renamed sanitized API workflow excludes nodes 154 and 166 and selects node 165 as the sole canonical edited-image output.
- [ ] #2 Every supported image-edit request value is applied or rejected before submission through the Image_Generation validation boundary.
- [ ] #3 Successful results use the existing Image_Generation attachment storage and generation-metadata contract, with no Video_Generation storage path.
- [ ] #4 Repository artifacts and history contain no prompt text, raw export artifact, or source identity from the workflow source.
- [ ] #5 Focused workflow, adapter, validation, and storage tests pass.
<!-- AC:END -->
