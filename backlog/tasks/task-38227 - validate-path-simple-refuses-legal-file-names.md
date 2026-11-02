---
id: TASK-38227
title: 'validate_path_simple refuses legal file names'
status: To Do
assignee: []
created_date: '2026-10-30 18:43'
labels:
  - security
  - bug
dependencies: []
references:
  - tldw_chatbook/Utils/path_validation.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
validate_path_simple refuses file names that are legal on every supported OS, such as `Q3 P&L; final.json`, because it rejects shell metacharacters in names that never reach a shell. It exists on dev today in every importer that calls it; Roleplay frame B2b's one Import picker keeps the refusal (now naming the file) and does not fix it. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Legal file names with &, ; and spaces pass validation wherever no shell is involved
- [ ] #2 Traversal and the inputs the validator exists to refuse are still refused, pinned by tests
<!-- AC:END -->
