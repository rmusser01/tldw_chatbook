---
id: TASK-32011
title: Recover fresh-install data storage from unsafe shared ancestors
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 00:30'
updated_date: '2026-09-08 01:10'
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
- [x] #6 A one-off Linux CI workflow validates the exact PR head against dev on supported Python versions and retains test evidence.
- [x] #7 Concurrent first starts share one private interprocess lock across default-root and profile creation, so permission repair cannot create conflicting roots.
- [ ] #8 Environment-derived existence probes use non-resolving central validation; prompt exports honor fallback and explicit roots; labeled CI reruns on subsequent PR commits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/127-fresh-install-private-data-root-recovery.md
Reason: Adds durable default-data selection under ADR-029; the review follow-up records interprocess coordination in the same ADR.

1. Record selection and conflict semantics; add failing fresh-install, restart, override, and unsafe-path regressions.
2. Implement secure durable recovery and read-only Settings display without weakening path guards.
3. Rebase on current dev and reproduce Qodo findings with real concurrent processes and named-profile export tests.
4. Serialize default-root selection and profile creation, validate lexical probes, centralize the fallback name, and fix export root selection and SQLite test transactions.
5. Rerun targeted storage and architecture checks, lint/format and independent review.
6. Rerun the labeled one-off Linux workflow automatically on synchronize, address review threads, record evidence, and merge the validated PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fresh installations now recover automatically from a missing conventional data root rejected for shared-writable .local/share ancestors by creating ~/.tldw_cli-data with the existing private directory guard. Directory presence retains the selection across processes and permission repairs. Explicit overrides, existing conventional entries, ambiguous roots, inaccessible paths, unsafe HOME, and symlinks retain refusal or explicit-selection behavior. No shared ancestor chmod, weakened guard, or implicit migration was introduced.

Settings storage diagnostics now use the same read-only selected-root helper. This integration was added to acceptance criteria after review identified its independent conventional-path calculation. Source changes are limited to config.py and settings_screen.py; regression coverage is in Tests/test_database_path_privacy.py and Tests/UI/test_profile_owned_settings_paths.py. README documents recovery and conflict handling. ADR-127 (backlog/decisions/127-fresh-install-private-data-root-recovery.md) records the storage decision under ADR-029; the design/plan is Docs/superpowers/specs/2026-09-07-fresh-install-data-root-recovery.md.

Verification: 10 new recovery/conflict/config-import cases failed against the original resolver; the Settings fallback display case failed before its fix. Final targeted command: .venv/bin/python -m pytest Tests/test_database_path_privacy.py Tests/test_user_data_dir_isolation.py Tests/test_config_private_bootstrap.py Tests/Utils/test_private_paths.py Tests/DB/test_private_sqlite.py Tests/UI/test_profile_owned_settings_paths.py Tests/test_logging_private_files.py -q --tb=short -> 427 passed, 1 Windows-only skip. Fresh subprocess config imports write and reopen a private SQLite row across restart and ancestor permission repair. An actual isolated tldw-cli --help invocation exits 0, creates the fallback, keeps the shared ancestor at 0775, and leaves the conventional root absent. Verification ran on macOS POSIX; native Linux execution was not available.

Architecture validation: 14 checks pass, including production profile-path inventory and its CLI. One existing census check fails because its unrestricted rglob reads an untracked, non-UTF-8 joblib fixture inside .claude/worktrees/internal-prompts-p1/.venv; this is unrelated to the patch and was not changed. Ruff passes for the changed storage test file, both test files pass formatting, and baseline comparison of all four changed Python files finds zero new lint findings or formatting deviations. Whole-file lint/format is not clean because config.py and settings_screen.py already carry unrelated debt. git diff --check passes for this change. Independent scoped code review found no remaining actionable findings.

Limitations: recovery applies to absent default data roots, not existing-data migration, unsafe HOME, or config-directory failures. The legacy Helper_Scripts/Prompts/Prompts_Dump.py compatibility-constant consumer remains outside normal runtime root resolution. No full-suite sweep, release, or deployment was performed.

Follow-up: prepare an isolated PR against current origin/dev and run one-off Linux CI at the PR head. Reapplied only this task patch; README and Settings context were adapted to the newer dev layout.

PR #2495 targets dev from isolated branch codex/fresh-install-private-storage. One-off Ubuntu 24.04 run https://github.com/rmusser01/tldw_chatbook/actions/runs/34174533166 passed at cff48b34d14eb0dee7fe083f4e6a6ad9767d6f71 on Python 3.11, 3.12, and 3.13: 437 tests passed and 4 platform-only tests skipped in each job. Each job also launched the installed tldw-cli under ancestor modes 0775 and 0755 and confirmed private fallback retention. JUnit artifacts were uploaded for all three interpreters. This supplies the previously missing native Linux evidence. The isolated current-dev local targeted run passed 439 tests with 2 Windows-only skips.

Qodo follow-up after rebase onto dev 511044368: all six comments addressed. A stable private HOME-level interprocess lock spans default-root selection and profile creation; a deterministic two-process test reproduces the old permission-repair race and verifies OS lock contention with the fix. Lexical existence probes use central validation without resolving symlinks. The fallback directory has a named constant. Prompt exports now honor the selected base while preserving explicit profile names and both database filenames, superseding the earlier legacy-helper limitation. SQLite regression writes use a transaction context. The opt-in Linux workflow also runs on synchronize while its evidence label remains. ADR-127 and the design plan document coordination.

Restored-source verification: 448 targeted tests passed, 2 Windows-only skips; three changed/new storage test files pass Ruff lint and formatting. Independent review of the restored code and new regression files found no actionable issues. Updated Linux matrix evidence will be recorded after publishing this revision.

All 15 profile-owned path architecture checks pass in the isolated worktree after the prompt helper census update.
<!-- SECTION:NOTES:END -->
