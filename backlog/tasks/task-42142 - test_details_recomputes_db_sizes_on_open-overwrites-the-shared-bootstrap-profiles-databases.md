---
id: TASK-42142
title: "test_details_recomputes_db_sizes_on_open overwrites the shared bootstrap profile's databases"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 19:11
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/UI/test_library_shell.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the consoletests stream (round 18). `Tests/UI/test_library_shell.py` is a bootstrap-profile file (`pytestmark` at line 214), so its tests share one profile per process. `test_details_recomputes_db_sizes_on_open` (line 6925) writes NUL-filled bytes (2048, 4096 and 5120 of them) over the files `get_prompts_db_path()`, `get_chachanotes_db_path()` and `get_media_db_path()` resolve to (lines 6956-6960, and the prompts WAL again at 6999), and never restores them. A later bootstrap-profile test in the same process that opens ChaChaNotes then fails with `CharactersRAGDBError ... file is not a database`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test no longer changes the shared profile's database files: it uses paths of its own, or restores each file it overwrites, also when it fails
- [x] #2 A test that opens ChaChaNotes, run in the same process right after it (`-p no:xdist`), passes; it is red before the change
- [x] #3 `Tests/UI/test_library_shell.py` still passes in its lane
<!-- AC:END -->

## Implementation Plan

1. Show red first: the zeroing node then a ChaChaNotes-opening bootstrap test (`Tests/UI/test_console_streaming_default_screen.py`) serially, `-p no:xdist`.
2. Point the test's three DB-path getters (`tldw_chatbook.config.get_*_db_path`, read at call time by `DBStatusManager._collect_db_sizes`) at files under the test's own `tmp_path` via `monkeypatch`, so the shared profile is never written.
3. Re-run the pair (green) and the file at -n 4.

## Implementation Notes

- **Fix.** `test_details_recomputes_db_sizes_on_open` now takes `tmp_path` and `monkeypatch` and points `tldw_chatbook.config.get_prompts_db_path`, `get_chachanotes_db_path` and `get_media_db_path` at files under its own `tmp_path`. `DBStatusManager._collect_db_sizes` imports those getters at call time, so the Details line reads the test's files, and the shared bootstrap profile's databases are never written. The real-home guard and the `mkdir`s went with the real paths. The assertions are unchanged; they can only pass on the sizes of the test's own files (3.0, 4.0, 5.0 and then 8.0 KB).
- **Red/green (AC#2).** `Tests/UI/test_console_streaming_default_screen.py` (bootstrap profile; it opens ChaChaNotes at `get_chachanotes_db_path()`), run right after the node, `-p no:xdist`:
  - base: 1 failed, 1 passed. The ChaChaNotes test failed with `CharactersRAGDBError: Database initialization failed: ... tldw_chatbook_ChaChaNotes.db': file is not a database`. It passes alone (1 passed). Load 57.
  - with the fix: 2 passed (load 38).
- **The file (AC#3).** Whole file at `-n 4`: 867 passed, 2 failed in 28 min (load 42-54). `test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back` is in `scripts/library_shell_push_deselect.txt` (a known product defect). `test_library_onboarding_back_requires_fresh_empty_evidence_and_focuses_import` (a focus assertion) passed twice alone right after (load 48), so it was load. Its lane is the push-to-dev library-shell shard job, which runs this file serially by node.
- Files: `Tests/UI/test_library_shell.py`.
