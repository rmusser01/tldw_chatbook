---
id: TASK-18801
title: >-
  Three manifest-boundary tests in test_summarization_diagnostic_privacy are red
  on origin/dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-18 23:52'
updated_date: '2026-09-05 20:20'
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
- [x] #2 Whichever of the checked-in inventory or the recorded fixture SHA is stale is identified and the fix explains which drifted and why
- [x] #3 Tests/LLM_Calls/test_summarization_diagnostic_privacy.py runs green as a whole
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the three reproduced manifest-boundary failures on the rebased review branch. 2. Wait for current Console/Library moves to settle, review every changed inventory owner and exact diagnostic statements under the existing inventory rules, then reconcile the checked manifest without altering privacy classification or limits. 3. Independently compute checked and generated normalized hashes and update only the stale boundary values; retain unrelated-drift, schema, summary and owner-mismatch mutant guards. 4. Run the complete summarization privacy module, inventory verification and scoped static checks; record original and reviewed final manifests. ADR required:no. ADR path:N/A. Reason: routine governed test-evidence reconciliation following existing TASK16213 pattern, no diagnostic policy or runtime boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled the rebased review branch without changing privacy assertions or diagnostic policy. The fixture still pinned 8f2ab91f... from an older unrelated-owner inventory. Reviewed current drift with --statements against manifest commit717f1951da: exactly ten identical warning statements moved from ChatScreen to retrieval.py (same statement-digest multiset, levels, arguments and exception handling); no new statement or sink. Rebuilt only those two owner rows. Independent checked/generated normalized projections both equal ac5cd5bf7bc9d5f35d80fd71a78953ea96cd6cc60fced84b2e4f60c332bc04f1. Totals unchanged:584owners,1336TASK492calls,7615TASK494calls,11sinks. Updated only the two fixture hashes; all257 complete summarization privacy tests pass in128.81s, including unrelated-drift, summary, owner/digest-schema and unreconciled-digest mutant guards. XML:/private/tmp/tldw-18801-summary-privacy-reconciled.xml. ADR not required: governed evidence reconciliation, no new boundary. AC1 remains unchecked because clean origin/dev has not received this draft branch; no merge or upstream-green claim.
<!-- SECTION:NOTES:END -->
