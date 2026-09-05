---
id: TASK-18801
title: >-
  Three manifest-boundary tests in test_summarization_diagnostic_privacy are red
  on origin/dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-18 23:52'
updated_date: '2026-09-05 23:40'
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
- [x] #4 Virtualenv exclusion tests retain exact owned-source and sink assertions with the upstream diagnostic summary schema
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the post-second-rebase baseline:322passes5failures in the two complete files.
2. Review the full inventory delta against codex/dev-test-review-before-second-rebase-20260905 using scanner statement multisets and source/sink inspection; verify every added source is byte-identical to origin/dev. Record the upstream Meetings classification without calling its exception diagnostics metadata-only.
3. Independently rebuild and normalize checked/generated inventories, then replace only the two stale manifest hashes and add the missing task_31551_calls zero to the exact virtualenv fixture summary. Preserve all negative mutant and scope assertions.
4. Run both complete files, inventory verification, scoped lint/format, independent review and save the draft checkpoint.
ADR required: no
ADR path: N/A
Reason: Governed test-evidence reconciliation of existing upstream inventory contracts; no diagnostic policy, ownership or security boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled the rebased review branch without changing privacy assertions or diagnostic policy. The fixture still pinned 8f2ab91f... from an older unrelated-owner inventory. Reviewed current drift with --statements against manifest commit717f1951da: exactly ten identical warning statements moved from ChatScreen to retrieval.py (same statement-digest multiset, levels, arguments and exception handling); no new statement or sink. Rebuilt only those two owner rows. Independent checked/generated normalized projections both equal ac5cd5bf7bc9d5f35d80fd71a78953ea96cd6cc60fced84b2e4f60c332bc04f1. Totals unchanged:584owners,1336TASK492calls,7615TASK494calls,11sinks. Updated only the two fixture hashes; all257 complete summarization privacy tests pass in128.81s, including unrelated-drift, summary, owner/digest-schema and unreconciled-digest mutant guards. XML:/private/tmp/tldw-18801-summary-privacy-reconciled.xml. ADR not required: governed evidence reconciliation, no new boundary. AC1 remains unchecked because clean origin/dev has not received this draft branch; no merge or upstream-green claim.

Post-second-rebase qualification at ed53b51793: inventory rebuild matches exactly (589 owners,1336 TASK492 calls,30 upstream TASK31551 calls,7615 TASK494 calls,12 sinks). Complete inventory+summary privacy selection:322passed5failed463.74s. Three failures are this task recurring after upstream inventory changed: normalized checked hash caa76e94acdbf3d61961e90bcfe307c21ed5c09bfc061d7c054ba497ff817684 versus prior ac5cd5bf7bc9d5f35d80fd71a78953ea96cd6cc60fced84b2e4f60c332bc04f1. The two other failures are virtualenv-exclusion fixture summary dictionaries omitting the new upstream task_31551_calls:0 field. AC3 is reopened for the current tree. No digest or privacy guard was changed in this qualification. Next: review upstream owner/sink delta and reconcile this governed boundary, preserving negative mutants. XML:/private/tmp/tldw-rebased-diagnostic-qualification.xml.

Second-rebase reconciliation: all 584 previous owner rows unchanged; five upstream Meetings owners add 30 TASK-31551 calls and snapshot_store adds two private stream sites. All six sources match origin/dev. Independent normalization reproduced the old hash and both checked/fresh rebuilt inventories produce caa76e94acdbf3d61961e90bcfe307c21ed5c09bfc061d7c054ba497ff817684. Replaced only two fixture hashes and the missing zero summary field; no exclusions, diagnostics policy or negative controls changed. Both complete files pass 327 tests in 409.28s (17 existing dependency/source warnings), XML /private/tmp/tldw-rebased-diagnostic-repaired.xml. Scoped Ruff and changed-range formatting pass; unrelated existing whole-file formatting drift remains. Independent scoped review clear. Full rationale: backlog/docs/diagnostic-rebase-reconciliation-2026-09-05.md. AC1 remains open: draft branch has not been merged into clean origin/dev. ADR not required: existing evidence boundary reconciliation.
<!-- SECTION:NOTES:END -->
