---
id: TASK-1502
title: 'Wizard RAG step: hide disabled model list when deps missing; fix backtick copy'
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: a long disabled embedding-model list renders under the 'not installed' message (reads as breakage); copy shows literal backticks in a TUI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No model list renders when embeddings deps are missing
- [ ] #2 Install copy renders without literal backticks
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Model RadioSet display=False when embeddings deps missing (was: disabled wall); copy quotes the extras package plainly instead of markdown backticks. Test asserts hidden list + no backtick.
<!-- SECTION:NOTES:END -->
