# Eval recovery CI repair implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; verify the existing regression before and after the smallest fix.

**Goal:** Repair PR #3287's temporary-directory collision and audit its unfinished checks.
**Architecture:** Preserve the private recovery fixture and every production guard. Give the selector regression a distinct test-owned directory.
**Tech Stack:** Python 3.12, pytest, native Windows fixture runner, GitHub Actions.
**Spec:** backlog/tasks/task-38268 - Fix-eval-recovery-temporary-directory-collision-after-portable-merge.md

ADR required: no
ADR path: N/A; existing ADR-126 applies.
Reason: test-only namespace correction with unchanged recovery policy.

## Constraints and review focus

Keep all TEMP/TMP/TMPDIR cases, actual tempfile discovery, secret filtering, and native fixture ownership. Run the exact nine-file eval-recovery selection; no full-suite sweep. Report pending CI explicitly. Do not merge the follow-up automatically.

## Task 1: Restore the selector regression and audit CI

Files: Tests/Backup_Recovery/test_profile_launch_environment.py; task record; lessons-testing-evidence.md.

- [x] Reproduce the three FileExistsError failures with the autouse fixture active.
- [x] Change only the test-owned directory from private-temp to selected-temp.
- [x] Run the five-case module, then the exact workflow's nine test files with --timeout=180.
- [x] Run scoped Ruff/format and diff checks; review the full change.
- [x] Record results, inspect every remaining PR #3287 check, and publish a follow-up PR.

## Added scope: Native AppData fixture isolation

The full native test discovered that HOME redirection leaves ConfigFileStorage's Windows APPDATA selector pointing at ambient pet state. Isolate APPDATA and LOCALAPPDATA in the existing per-test recovery fixture. Production recovery policy is unchanged; existing ADR-126 applies.

- [x] Add a parameterized regression requiring both native selectors to name pinned private paths under tmp_path while generic restart filtering continues to exclude them; observe it fail first.
- [x] Create two private fixture-owned native application-data roots and redirect the selectors with monkeypatch.
- [x] Verify all seven launch-environment cases, the complete backup/reopen case, and the full CI selection. Keep host profile data outside test archives.

## Verification and remaining gates

- Source e2ec8715e3850551a75361cf68531403bb82bd57: hosted Ubuntu 85 passed; macOS 85 passed; Windows 84 passed, 1 unsupported-FIFO skip. [Matrix run](https://github.com/rmusser01/tldw_chatbook/actions/runs/41257072705).
- Final native local nine-file selection: 81 passed, 4 platform capability skips, 1061.94 seconds; actual host pet-state bytes unchanged.
- Seven focused environment cases, scoped Ruff/format/whitespace and independent review passed. Production recovery checks/timeouts remain unchanged. Full repository suite was not run.
- [Draft follow-up #3342](https://github.com/rmusser01/tldw_chatbook/pull/3342) remains In Progress pending 34 queued checks and a base update when preparing to merge. Original PR #3287 still has 36 queued checks. No merge/auto-merge performed.
- The verification-record commit changes only documentation; all tested source bytes match e2ec8715e3.

## Rebase and ready-for-review continuation

The owner requested rebasing PR #3342 onto dev and resolving outstanding CI before leaving draft. Rebased both commits without conflicts onto dev 77718ebebc; git range-diff reports both original patches unchanged. Focused native environment tests: 7 passed. The earlier three-platform results above remain historical evidence for the pre-rebase source; current-head CI must qualify the integrated tree.

Keep the PR in draft while reviewing the normal pull_request CI runs and available review threads. Retire only superseded runs for this PR if their always-run final gate holds its concurrency slot; never dispatch the required check by hand. No merge or auto-merge is authorized. Live final CI results will be recorded in the PR description so recording them does not create another source head.

## Owner-approved explicit runner trial

ADR required: yes (existing ADR-103 amendment).
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: owner approved testing the already-executing Ubuntu 24.04 image for the required workflow's queued Linux jobs. Keep all gate/capacity contracts intact.

- [x] Change the five existing Derived Artifacts runner assertions and observe RED against ubuntu-latest (7 intended failures, 82 passed).
- [x] Pin the workflow's nine runs-on values to ubuntu-24.04 and verify the existing contract suite (150 passed, 186.59s); parsed workflow otherwise identical, independent review clean.
- [ ] Confirm current-head hosted allocation, all required checks and available review feedback before leaving draft.

The runner trial passed the existing workflow-contract suite, 1,242-entry UI census and both Backlog guards. Scoped Ruff lint passes; the five assertion literals introduce no formatting drift relative to the already-unformatted baseline. Dev advanced to 2561efd658 during verification; rebase the qualified patch onto that base before publication and use current-head hosted CI for the integrated result.
