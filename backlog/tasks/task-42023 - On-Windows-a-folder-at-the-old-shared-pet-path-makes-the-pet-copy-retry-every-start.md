---
id: TASK-42023
title: "On Windows a folder at the old shared pet path makes the pet copy retry every start"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:39
labels:
- follow-up
- pets
- windows
priority: low
references:
- tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-42000 (round 17), not run. If a directory sits at the old shared pet file's path, the Windows open raises `PermissionError`, which the upgrade copy treats as a failed read, so it retries on every start instead of treating it as nothing to copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A directory at the old shared path is treated as nothing to copy on every platform, including Windows, where the open raises `PermissionError`
- [x] #2 A regular file that cannot be read (for example locked by another process) still counts as a failed read: nothing is written, no marker, and the next start retries, as `_read_shared_pets` documents
- [x] #3 Tests simulate the Windows `PermissionError` for both cases on every platform, the directory case red before the fix
<!-- AC:END -->

## Implementation Plan

1. Add a test in `Tests/Backup_Recovery/test_operational_owners.py`, parametrized over a directory and a regular file at the old shared path, that makes `os.open` of that path raise `PermissionError` (as Windows does for both) on every platform, scoped to the store's start.
   - Directory: the store starts with no pets and writes the seed marker (red before the fix).
   - Locked file: the start raises, nothing is written, and the next start (without the refusal) copies the pets.
2. In `_read_shared_pets`, when the open fails and the path is a directory, count it as nothing to copy; any other failure on an existing file still raises.
3. Run the pet tests in `test_operational_owners.py` and `Tests/Widgets/test_tamagotchi*.py`.

## Implementation Notes

**Approach.** `_read_shared_pets` (`Widgets/Tamagotchi/tamagotchi_storage.py`) now counts a failed open as nothing to copy when the old shared path is a directory (`os.path.isdir`), as it already did for a link (`ELOOP`). On POSIX the open of a directory succeeds and `fstat` already said "not a file"; on Windows the open raises `PermissionError`, which used to propagate as a failed read. Any other failure on a path that is not a directory still raises, so a locked or unreadable file is a failed read: nothing is written, no marker, and the next start retries.

**Tests** (`Tests/Backup_Recovery/test_operational_owners.py`, run by the Backup_Recovery owner-suites lane): `test_a_windows_refusal_at_the_old_shared_pet_path[directory|locked_file]`. A scoped `monkeypatch.context()` makes `os.open` of the shared path raise `PermissionError(EACCES)` on every platform, and delegates every other open.
- `[directory]`: the store starts with no pets and writes `tamagotchi_pets.seeded` (AC#1).
- `[locked_file]`: the start raises `PermissionError`, the profile folder stays empty (no pet file, no marker), and the next start without the refusal copies the pets (AC#2).
- The existing `test_a_failed_pet_copy_writes_nothing_and_the_next_start_retries[unreadable]` (a real `chmod 0` file, POSIX only) still covers the unreadable-file refusal.

**Evidence** (`-n 4`, load 25-28).
- Red before the fix: `[directory]` failed with `PermissionError: [Errno 13] Permission denied: '.../home/.config/tldw_chatbook/tamagotchi_pets.json'`; `[locked_file]` passed (1 failed, 1 passed).
- Green: `test_operational_owners.py` and `Tests/Widgets/test_tamagotchi*.py` (four files), 196 passed.
- `ruff check --select F` is clean on the touched source. The test file's existing F841 (line 599) and both files' `ruff format` differences are at base and not in this diff.
- No size ratchet lists either file. No User Guide change: the settings page says the shared file is copied "if there is one", and a directory is not one.
