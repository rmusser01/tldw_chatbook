---
id: TASK-18801
title: >-
  Three manifest-boundary tests in test_summarization_diagnostic_privacy are red
  on origin/dev
status: To Do
assignee: []
created_date: '2026-08-18 23:52'
labels:
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py has three failing tests on a clean origin/dev checkout at 7d87686a3, unrelated to any feature branch:

  test_manifest_boundary_changes_only_summarization_owner_diagnostics
  test_manifest_boundary_rejects_owned_digest_schema_changes
  test_manifest_boundary_rejects_unreconciled_owned_digest

The first asserts that a normalized projection of the checked-in diagnostic inventory hashes to the SHA recorded in the review fixture, and it does not:

  AssertionError: checked inventory changed outside the two summarization owners
  assert 'b0187e7972ac...85edefc4ed7ee' == '8b0633e98e95...05bfd61aac9de'

The other two fail as a consequence -- both call the first test as their control before applying their mutant.

Reproduced on a dedicated detached worktree of origin/dev with nothing else applied: 3 failed, 254 passed. The same three, and only those three, appear when running the file from a feature branch, so branches touching tldw_chatbook/LLM_Calls/ currently inherit a red gate they did not cause. Either the checked-in inventory JSON or the fixture SHA needs regenerating and reconciling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three manifest-boundary tests pass on a clean origin/dev checkout
- [ ] #2 Whichever of the checked-in inventory or the recorded fixture SHA is stale is identified and the fix explains which drifted and why
- [ ] #3 Tests/LLM_Calls/test_summarization_diagnostic_privacy.py runs green as a whole
<!-- AC:END -->
