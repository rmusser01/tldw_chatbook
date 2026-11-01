---
id: TASK-38268
title: Fix eval recovery temporary directory collision after portable merge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-31T23:49:00Z'
updated_date: '2026-11-01T03:30:00Z'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The merged portable bundle introduced an autouse private temporary directory that collides with the existing launch-environment regression on Ubuntu and macOS. Restore meaningful coverage and audit the remaining merged PR CI results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three native temporary selectors reach and pass their launch-environment assertions with the autouse private fixture active.
- [x] #2 The targeted eval-recovery workflow and remaining PR checks are verified or their unresolved results are explicitly recorded.
- [x] #3 Windows recovery tests and their child processes use owned APPDATA and LOCALAPPDATA sandboxes so ambient installed pet state cannot enter a test backup.
- [ ] #4 The required workflow executes its unchanged gate on the owner-approved Ubuntu 24.04 runner and current-head verification is resolved before leaving draft.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing ADR-103; no duplicate ADR).
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: owner-approved explicit Linux runner selection for the required workflow, preserving its gate and capacity contracts. Existing ADR-126 still governs unchanged recovery authority.
1. Preserve both recovery fixture fixes and the clean rebase onto dev 77718ebebc.
2. Record the observed ubuntu-latest queue and the owner's approved ubuntu-24.04 trial in ADR-103 before changing the workflow.
3. Update only existing Derived Artifacts runner assertions, observe them fail against the floating label, then pin all nine workflow jobs to ubuntu-24.04 and verify them.
4. Preserve test selectors, shard counts/parallelism, dependencies, permissions, timeout budgets and fail-closed aggregation; run the existing workflow-contract tests and relevant static guards.
5. Push the qualified change, retire only this PR's superseded runs, and inspect all current-head CI/review feedback before marking ready. No merge or auto-merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the FileExistsError in all three original PR #3287 eval-recovery jobs by giving the selector test its own selected-temp directory. Native qualification also found ambient Windows pet state entering the full backup fixture through APPDATA; the shared recovery fixture now creates private APPDATA/LOCALAPPDATA roots. Both new isolation regressions failed before the fix and pass afterward. Generic restart filtering and all production recovery guards remain unchanged.

Verification on source commit e2ec8715e3850551a75361cf68531403bb82bd57: hosted nine-file eval-recovery lane passed on Ubuntu (85), macOS (85), and Windows (84 passed, one unsupported-FIFO skip). Run: https://github.com/rmusser01/tldw_chatbook/actions/runs/41257072705. Final local same-lane run: 81 passed, four native capability skips, 1061.94s; real pet-state bytes unchanged. Scoped Ruff/format/whitespace checks and independent code review passed. Full repository suite not run under opt-in policy.

The first optional-heavy local environment timed out at the unchanged 90s cold-start budget. Clean qualification then correctly refused a shared source folder and ambient pet relocation; private native setup and AppData fixture isolation resolved those failures without changing runtime code or timeouts. Generated copies from the failed ambient-state test were removed only from its verified task-owned temporary subtree.

Changed test files: Tests/Backup_Recovery/conftest.py and test_profile_launch_environment.py. Added plan and incident lessons. ADR required: no new ADR; existing backlog/decisions/126-complete-local-backup-and-recovery.md applies to unchanged recovery authority/private storage policy.

Follow-up: https://github.com/rmusser01/tldw_chatbook/pull/3342. The initial draft had 34 queued checks and was behind dev; the owner subsequently requested its rebase and qualification before leaving draft. Both patches were rebased unchanged onto dev 77718ebebc with no conflicts, and all seven focused native environment tests passed. Keep In Progress pending current-head CI and review; final live results are maintained in the PR description. No merge or auto-merge was requested or performed. Historical source qualification above is retained and is not a claim for the newly integrated runtime.

Owner-approved runner trial (existing ADR-103 amendment): pin all nine Derived Artifacts runs-on declarations to ubuntu-24.04 and update only five existing runner assertions. RED: seven intended label mismatches, 82 passed. GREEN: both existing workflow-contract files, 150 passed in 186.59s. Complete parsed workflow is identical apart from labels; all 1,242 UI entries, 32 shards, parallelism, dependencies, permissions, timeouts and fail-closed aggregation are preserved. UI census and Backlog guards pass; independent focused review clean. Ruff lint passes, and the literal edits preserve the pre-existing formatter debt. Dev advanced to 2561efd658 during qualification; publish the rebased patch and keep draft until current-head hosted verification resolves. The explicit-label scheduling result remains unproven.

<!-- SECTION:NOTES:END -->
