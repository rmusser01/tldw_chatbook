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
- [ ] Run the five-case module, then the exact workflow's nine test files with --timeout=180.
- [ ] Run scoped Ruff/format and diff checks; review the full change.
- [ ] Record results, inspect every remaining PR #3287 check, and publish a follow-up PR.

## Added scope: Native AppData fixture isolation

The full native test discovered that HOME redirection leaves ConfigFileStorage's Windows APPDATA selector pointing at ambient pet state. Isolate APPDATA and LOCALAPPDATA in the existing per-test recovery fixture. Production recovery policy is unchanged; existing ADR-126 applies.

- [ ] Add a parameterized regression requiring both native selectors to name pinned private paths under tmp_path while generic restart filtering continues to exclude them; observe it fail first.
- [ ] Create two private fixture-owned native application-data roots and redirect the selectors with monkeypatch.
- [ ] Verify all seven launch-environment cases, the complete backup/reopen case, and the full CI selection. Keep host profile data outside test archives.
