---
id: TASK-505
title: Clear inherited lint findings in the citation verification scope
status: Done
assignee: []
created_date: '2026-07-24 16:50'
updated_date: '2026-07-24 16:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the eight pre-existing Ruff findings covered by the citation foundation's required broad lint command so the verification gate can pass without exclusions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The exact citation-foundation Ruff command reports zero findings
- [x] #2 Unused imports and locals are removed without changing behavior
- [x] #3 Test-only lambdas are expressed as named local functions
- [x] #4 Focused Console, RAG-scope, character-card, and subscription tests pass
- [x] #5 No citation production behavior is changed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact eight-finding Ruff output as RED evidence. 2. Apply only mechanical unused-import, unused-local, and lambda-to-def edits. 3. Run focused tests for each touched area and the exact broad Ruff gate. 4. Record no-ADR rationale, verification, acceptance criteria, and implementation notes before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed two unused imports and two unused assignment targets while retaining both calls, and replaced the four assigned test lambdas with behavior-equivalent named local functions. No citation logic or other production behavior changed. ADR required: no; ADR path: N/A; Reason: mechanical lint-only cleanup that changes no architecture, schema, storage, security, or runtime contract. RED evidence: the exact citation-foundation Ruff command reported eight inherited findings (two F401, two F841, four E731). Verification: the same broad Ruff command reports zero findings; focused Console history/edit-resend, RAG-scope, character-card paging, and subscription suites pass with 41 tests; all edited ranges pass Ruff format range checks; git diff --check passes. Whole-file Ruff formatting remains inherited debt in the five legacy files and was intentionally not applied because it would create a large out-of-scope formatting diff.
<!-- SECTION:NOTES:END -->
