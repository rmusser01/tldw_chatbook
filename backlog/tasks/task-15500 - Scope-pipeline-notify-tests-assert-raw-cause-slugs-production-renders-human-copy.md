---
id: TASK-15500
title: >-
  Scope-pipeline notify tests assert raw cause slugs; production renders human
  copy
status: To Do
assignee: []
created_date: '2026-08-11 19:48'
labels:
  - bug
  - rag
  - tests
  - pre-existing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four tests in Tests/RAG/test_scope_pipeline_enforcement.py assert that a raw cause slug ("deleted-items", "workspace-read-failed") appears in the user-facing notify string, but production has rendered human copy ("the scoped items have been deleted") since dev commit 6f3b9b6d3 (a Chat-side change). The tests encode the old contract, so they are red on dev and the real notify copy is unguarded. Decide which side is the contract and make one of them follow the other.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/RAG/test_scope_pipeline_enforcement.py passes on a clean dev checkout
- [ ] #2 The assertions check the copy the user actually sees, or the production seam is changed to emit what they check — with the choice recorded
- [ ] #3 The user-facing notify copy for each scope-failure cause is pinned by a test, so a future copy change cannot silently pass
<!-- AC:END -->
