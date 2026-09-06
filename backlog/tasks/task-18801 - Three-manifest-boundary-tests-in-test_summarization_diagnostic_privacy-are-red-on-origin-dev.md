---
id: TASK-18801
title: >-
  Three manifest-boundary tests in test_summarization_diagnostic_privacy are red
  on origin/dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-18 23:52'
updated_date: '2026-09-06 01:12'
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
1. Preserve fourth-rebase evidence: 144 passed and one stale manifest hash failure on dev 56376e1fc188938bf350c62d3a9f95e820b93c40.
2. Review the exact app.py/config.py diagnostic statement delta against codex/dev-test-review-before-fourth-rebase-20260906; verify all other owners, sink topology and classifications remain unchanged.
3. Independently reproduce the previous normalized hash and compute the current hash; replace only the two fixture pins, leaving all boundary and negative-control logic unchanged.
4. Run both complete diagnostic files, scoped static checks and independent review; record the fourth rebase and update the draft PR without claiming clean dev or full-suite completion.
ADR required: no
ADR path: N/A
Reason: Governed evidence reconciliation of existing upstream diagnostics; no runtime, privacy or ownership boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled the rebased review branch without changing privacy assertions or diagnostic policy. The fixture still pinned 8f2ab91f... from an older unrelated-owner inventory. Reviewed current drift with --statements against manifest commit717f1951da: exactly ten identical warning statements moved from ChatScreen to retrieval.py (same statement-digest multiset, levels, arguments and exception handling); no new statement or sink. Rebuilt only those two owner rows. Independent checked/generated normalized projections both equal ac5cd5bf7bc9d5f35d80fd71a78953ea96cd6cc60fced84b2e4f60c332bc04f1. Totals unchanged:584owners,1336TASK492calls,7615TASK494calls,11sinks. Updated only the two fixture hashes; all257 complete summarization privacy tests pass in128.81s, including unrelated-drift, summary, owner/digest-schema and unreconciled-digest mutant guards. XML:/private/tmp/tldw-18801-summary-privacy-reconciled.xml. ADR not required: governed evidence reconciliation, no new boundary. AC1 remains unchecked because clean origin/dev has not received this draft branch; no merge or upstream-green claim.

Post-second-rebase qualification at ed53b51793: inventory rebuild matches exactly (589 owners,1336 TASK492 calls,30 upstream TASK31551 calls,7615 TASK494 calls,12 sinks). Complete inventory+summary privacy selection:322passed5failed463.74s. Three failures are this task recurring after upstream inventory changed: normalized checked hash caa76e94acdbf3d61961e90bcfe307c21ed5c09bfc061d7c054ba497ff817684 versus prior ac5cd5bf7bc9d5f35d80fd71a78953ea96cd6cc60fced84b2e4f60c332bc04f1. The two other failures are virtualenv-exclusion fixture summary dictionaries omitting the new upstream task_31551_calls:0 field. AC3 is reopened for the current tree. No digest or privacy guard was changed in this qualification. Next: review upstream owner/sink delta and reconcile this governed boundary, preserving negative mutants. XML:/private/tmp/tldw-rebased-diagnostic-qualification.xml.

Second-rebase reconciliation: all 584 previous owner rows unchanged; five upstream Meetings owners add 30 TASK-31551 calls and snapshot_store adds two private stream sites. All six sources match origin/dev. Independent normalization reproduced the old hash and both checked/fresh rebuilt inventories produce caa76e94acdbf3d61961e90bcfe307c21ed5c09bfc061d7c054ba497ff817684. Replaced only two fixture hashes and the missing zero summary field; no exclusions, diagnostics policy or negative controls changed. Both complete files pass 327 tests in 409.28s (17 existing dependency/source warnings), XML /private/tmp/tldw-rebased-diagnostic-repaired.xml. Scoped Ruff and changed-range formatting pass; unrelated existing whole-file formatting drift remains. Independent scoped review clear. Full rationale: backlog/docs/diagnostic-rebase-reconciliation-2026-09-05.md. AC1 remains open: draft branch has not been merged into clean origin/dev. ADR not required: existing evidence boundary reconciliation.

Fourth-rebase qualification on dev 56376e1fc188938bf350c62d3a9f95e820b93c40: baseline 144 passed / 1 stale-hash failure. Reviewed exactly two added upstream app/config warnings with scanner statements; all other 587 owner rows, classifications and 12 sink files are unchanged. Independently reproduced the old caa76e94... pin and current 0a0c4b6dbe89debeacc1d1b662d2ec6275a7e98e082fd0325ad469d345a4c7eb; replaced only the two fixture hashes. Both complete diagnostic files pass 327 tests in 533.19s including unchanged negative controls, with 17 existing dependency/source warnings. XML: /private/tmp/tldw-fourth-rebase-diagnostic-repaired.xml. Canonical JSON formatting, scoped repair lint/format, diff checks and independent review pass. Detailed evidence: backlog/docs/diagnostic-rebase-reconciliation-2026-09-05.md. AC1 remains open because this draft has not been merged into clean origin/dev. ADR required: no; existing governed evidence reconciliation.
<!-- SECTION:NOTES:END -->
