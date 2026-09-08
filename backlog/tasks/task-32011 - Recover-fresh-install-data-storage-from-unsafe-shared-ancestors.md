---
id: TASK-32011
title: Recover fresh-install data storage from unsafe shared ancestors
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 00:30'
updated_date: '2026-09-08 00:45'
labels:
  - bug
  - storage
  - privacy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh installs can fail during configuration import when existing .local or share directories are writable by a group. Make default storage selection recover automatically without requiring contact with the reporter, changing shared ancestor permissions, weakening private-path validation, or hiding existing data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh installs with group- or world-writable .local or share ancestors automatically use a private home-level data directory without changing those ancestors.
- [x] #2 The selected fallback remains stable across processes and later permission repairs, while configured roots and existing default data are never silently bypassed.
- [x] #3 The original ownership, symlink, private-mode, and unsafe-home protections remain enforced for all selected storage.
- [x] #4 Real config-import and SQLite persistence regression tests cover fresh launch and restart, and targeted storage checks pass.
- [x] #5 Settings storage diagnostics show the selected default data location without creating directories.
- [ ] #6 A one-off Linux CI workflow validates the exact PR head against dev on supported Python versions and retains test evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/127-fresh-install-private-data-root-recovery.md
Reason: Adds a durable default-data location selection rule under the existing ADR-029 private-storage boundary.

1. Record selection and conflict semantics in ADR-127 and a focused design/plan.
2. Add failing fresh-install, restart, existing-root, override, and unsafe-path tests.
3. Add one default-data resolver that secures the conventional root or selects the home-level fallback only for a missing conventional root rejected as shared-writable.
4. Verify real config import and SQLite persistence across fresh processes, then run targeted storage and path-owner checks and lint/format checks.
5. Update storage documentation, review the diff, record evidence and complete the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fresh installations now recover automatically from a missing conventional data root rejected for shared-writable .local/share ancestors by creating ~/.tldw_cli-data with the existing private directory guard. Directory presence retains the selection across processes and permission repairs. Explicit overrides, existing conventional entries, ambiguous roots, inaccessible paths, unsafe HOME, and symlinks retain refusal or explicit-selection behavior. No shared ancestor chmod, weakened guard, or implicit migration was introduced.

Settings storage diagnostics now use the same read-only selected-root helper. This integration was added to acceptance criteria after review identified its independent conventional-path calculation. Source changes are limited to config.py and settings_screen.py; regression coverage is in Tests/test_database_path_privacy.py and Tests/UI/test_profile_owned_settings_paths.py. README documents recovery and conflict handling. ADR-127 (backlog/decisions/127-fresh-install-private-data-root-recovery.md) records the storage decision under ADR-029; the design/plan is Docs/superpowers/specs/2026-09-07-fresh-install-data-root-recovery.md.

Verification: 10 new recovery/conflict/config-import cases failed against the original resolver; the Settings fallback display case failed before its fix. Final targeted command: .venv/bin/python -m pytest Tests/test_database_path_privacy.py Tests/test_user_data_dir_isolation.py Tests/test_config_private_bootstrap.py Tests/Utils/test_private_paths.py Tests/DB/test_private_sqlite.py Tests/UI/test_profile_owned_settings_paths.py Tests/test_logging_private_files.py -q --tb=short -> 427 passed, 1 Windows-only skip. Fresh subprocess config imports write and reopen a private SQLite row across restart and ancestor permission repair. An actual isolated tldw-cli --help invocation exits 0, creates the fallback, keeps the shared ancestor at 0775, and leaves the conventional root absent. Verification ran on macOS POSIX; native Linux execution was not available.

Architecture validation: 14 checks pass, including production profile-path inventory and its CLI. One existing census check fails because its unrestricted rglob reads an untracked, non-UTF-8 joblib fixture inside .claude/worktrees/internal-prompts-p1/.venv; this is unrelated to the patch and was not changed. Ruff passes for the changed storage test file, both test files pass formatting, and baseline comparison of all four changed Python files finds zero new lint findings or formatting deviations. Whole-file lint/format is not clean because config.py and settings_screen.py already carry unrelated debt. git diff --check passes for this change. Independent scoped code review found no remaining actionable findings.

Limitations: recovery applies to absent default data roots, not existing-data migration, unsafe HOME, or config-directory failures. The legacy Helper_Scripts/Prompts/Prompts_Dump.py compatibility-constant consumer remains outside normal runtime root resolution. No full-suite sweep, release, or deployment was performed.

Follow-up: prepare an isolated PR against current origin/dev and run one-off Linux CI at the PR head. Reapplied only this task patch; README and Settings context were adapted to the newer dev layout.
<!-- SECTION:NOTES:END -->
