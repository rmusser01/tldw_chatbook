---
id: TASK-42172
title: >-
  A fresh profile with no config folder opens on 'Recovery required:
  configuration_unavailable'
status: Done
assignee:
  - '@claude'
created_date: '2026-11-21 05:30'
labels:
  - startup
  - backup-recovery
  - linux
priority: high
references:
  - tldw_chatbook/Backup_Recovery/launcher.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in a Linux UAT on dev 230205800c (2026-11-20, Debian 12, a new user account). On first launch the app opens on "Recovery required: configuration_unavailable" with only Backup & Restore. The launcher's startup preflight reads the config through the recovery reader. When `~/.config` or `~/.config/tldw_cli` does not exist, Linux's private-path walk raises PrivatePathError(missing_parent) instead of FileNotFoundError. The preflight treats only FileNotFoundError as "no config yet", so every new Linux install was refused before the config loader (which already treats missing_parent as a fresh profile) could create the profile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A profile with no config folder (missing ~/.config or ~/.config/tldw_cli) starts normally and creates its config
- [x] #2 Other private-path refusals (shared writable parent, unsafe parents) still stop startup with their existing reasons
- [x] #3 A test fails on the pre-fix code for both a real missing folder and the Linux missing_parent shape
- [x] #4 Verified on Linux: a new account with no ~/.config launches into the Console
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red test in Tests/Backup_Recovery/test_startup_unlock.py (real missing folders + forced missing_parent).
2. startup_preflight: treat PrivatePathError(missing_parent) like FileNotFoundError, as config.py's loader already does.
3. Re-launch on Linux with no ~/.config.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix (Backup_Recovery/launcher.py `startup_preflight`):** PrivatePathError with reason `missing_parent` returns `(None, None)`, the same as FileNotFoundError. `config.py`'s loader already treats `missing_parent` as a fresh profile and creates the folder under its own private-path checks. `shared_writable_parent` and every other reason keep their existing refusals.

**Test:** `Tests/Backup_Recovery/test_startup_unlock.py::test_fresh_profile_without_a_config_folder_starts_normally` covers two cases, real missing folders and a forced Linux `missing_parent`. Both returned `configuration_unavailable` on base. Inside pytest, macOS raised `missing_parent` for real missing folders too; a scratch-HOME probe outside pytest got FileNotFoundError. The rest of the file still passes (32/32).

**Live (Linux, 2026-11-20):** a new account with no `~/.config` launched straight into the Console in about 12 s and created `~/.config/tldw_cli/config.toml`. Before the fix it opened on "Recovery required: configuration_unavailable".
<!-- SECTION:NOTES:END -->
