---
id: TASK-42142
title: "test_details_recomputes_db_sizes_on_open overwrites the shared bootstrap profile's databases"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:33
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
- [ ] #1 The test no longer changes the shared profile's database files: it uses paths of its own, or restores each file it overwrites, also when it fails
- [ ] #2 A test that opens ChaChaNotes, run in the same process right after it (`-p no:xdist`), passes; it is red before the change
- [ ] #3 `Tests/UI/test_library_shell.py` still passes in its lane
<!-- AC:END -->
