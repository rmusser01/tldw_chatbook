---
id: TASK-38268
title: Fix eval recovery temporary directory collision after portable merge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-31T23:49:00Z'
updated_date: '2026-11-01T00:53:00Z'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A (existing ADR-126 recovery guards remain unchanged)
Reason: routine test-only namespace collision; no new runtime or trust boundary.
1. Reproduce all three FileExistsError failures from PR #3287 with native private fixture setup.
2. Give the launch-environment test its own temporary directory and retain all selector/secrets assertions.
3. Verify the affected module and exact eval-recovery workflow selection; run scoped lint and review the diff.
4. Audit all remaining PR #3287 check results, record unresolved checks, and publish a scoped follow-up PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the FileExistsError in all three original PR #3287 eval-recovery jobs by giving the selector test its own selected-temp directory. Native qualification also found ambient Windows pet state entering the full backup fixture through APPDATA; the shared recovery fixture now creates private APPDATA/LOCALAPPDATA roots. Both new isolation regressions failed before the fix and pass afterward. Generic restart filtering and all production recovery guards remain unchanged.

Verification on source commit e2ec8715e3850551a75361cf68531403bb82bd57: hosted nine-file eval-recovery lane passed on Ubuntu (85), macOS (85), and Windows (84 passed, one unsupported-FIFO skip). Run: https://github.com/rmusser01/tldw_chatbook/actions/runs/41257072705. Final local same-lane run: 81 passed, four native capability skips, 1061.94s; real pet-state bytes unchanged. Scoped Ruff/format/whitespace checks and independent code review passed. Full repository suite not run under opt-in policy.

The first optional-heavy local environment timed out at the unchanged 90s cold-start budget. Clean qualification then correctly refused a shared source folder and ambient pet relocation; private native setup and AppData fixture isolation resolved those failures without changing runtime code or timeouts. Generated copies from the failed ambient-state test were removed only from its verified task-owned temporary subtree.

Changed test files: Tests/Backup_Recovery/conftest.py and test_profile_launch_environment.py. Added plan and incident lessons. ADR required: no new ADR; existing backlog/decisions/126-complete-local-backup-and-recovery.md applies to unchanged recovery authority/private storage policy.

Follow-up: https://github.com/rmusser01/tldw_chatbook/pull/3342. The initial draft had 34 queued checks and was behind dev; the owner subsequently requested its rebase and qualification before leaving draft. Both patches were rebased unchanged onto dev 77718ebebc with no conflicts, and all seven focused native environment tests passed. Keep In Progress pending current-head CI and review; final live results are maintained in the PR description. No merge or auto-merge was requested or performed. Historical source qualification above is retained and is not a claim for the newly integrated runtime.
<!-- SECTION:NOTES:END -->
