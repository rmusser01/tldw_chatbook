---
id: TASK-25705
title: Reconcile diagnostic inventory and enforce the dev required gate
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 15:54'
updated_date: '2026-08-30 16:18'
labels:
  - diagnostics
  - ci
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the repository-wide persistent-diagnostic contract after merged diagnostic-bearing changes drifted from its generated inventory, and close the branch-protection bypass that allowed the drift to merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every newly introduced or changed persistent diagnostic statement is reviewed against ADR-029 and contains no private user-authored content.
- [x] #2 The canonical persistent-diagnostic inventory and its dependent summarization boundary fixture reproduce from current sources.
- [x] #3 The focused inventory and summarization privacy tests pass without changing production behavior unless review identifies an unsafe diagnostic.
- [x] #4 ADR-103 documents the merge incident and the stricter current-base/admin enforcement decision.
- [x] #5 The dev branch requires the existing derived-artifacts context on the latest base and applies that requirement to administrators.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the inventory drift and inspect every changed diagnostic statement against ADR-029.
2. Regenerate the canonical inventory and reconcile its dependent summarization fixture without changing runtime behavior unless privacy review requires it.
3. Amend ADR-103 with the observed admin/latest-base bypass and the stricter protection decision.
4. Run the focused inventory, privacy, and submitted-log regression matrix plus repository hygiene checks.
5. Enable admin enforcement and strict latest-base checks for the existing dev required context, then verify the live protection state.
6. Complete task hygiene and take the recovery branch through PR review and merge.

ADR required: yes
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: This changes the long-lived enforcement semantics of the dev required-check boundary after a real bypass; ADR-029 remains the governing privacy contract for diagnostic review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed every diagnostic introduced by the merged workspace/persona changes against ADR-029. Replaced raw policy entries, workspace/persona identifiers, exception messages, and persisted tracebacks with fixed labels plus safe type/count metadata; control flow and user-visible error handling are unchanged.

Regenerated Docs/security/production-diagnostic-inventory.json and reconciled the summarization privacy-boundary hash. Removed three stale historical diagnostic guard rows whose source statements had already been retired. Amended ADR-103 and the testing-evidence lesson with the reproduced PR #2228 admin/stale-base bypass. Live dev protection now requires the existing derived-artifacts context with strict/latest-base and admin enforcement enabled; force-push policy was not changed.

Verification: after the final `dev` rebase, the canonical checker reports 547 owners / 1,278 TASK-492 calls / 7,397 TASK-494 calls / 8 sink files with no drift; 66 focused runtime/privacy tests pass; 323 inventory/privacy pin cases pass; 5 Qodo-focused review regressions pass; the strict-CP1252 submitted-log matrix reports 738 passed and 7 expected skips; git diff --check passes. ADR required: yes; ADR-103 amended, with ADR-029 governing diagnostic privacy.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task was renumbered from `TASK-24653` to `TASK-25705` after rebasing
exposed a concurrent collision. The older “Network TLS trust policy (corp DPI)”
task was created at 2026-08-29 22:51 and keeps `TASK-24653` under the
TASK-19601 owner rule; this recovery task was created at 2026-08-30 15:54 and
therefore moved. A live sweep of every remote ref and local worktree confirmed
`TASK-25705` was unused at renumbering time. ADR-103 and the testing-evidence
lesson were updated to follow this task’s new id.

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Canonical inventory checker and focused/submitted-log regression matrices pass.
- [x] #2 Diff hygiene and changed-line static review complete without new issues.
- [x] #3 ADR-103, the testing-evidence lesson, and task documentation are current.
- [x] #4 Self-review confirms production changes are limited to metadata-only diagnostics.
- [ ] #5 PR review feedback is addressed and the required merge gate is green.
<!-- DOD:END -->
