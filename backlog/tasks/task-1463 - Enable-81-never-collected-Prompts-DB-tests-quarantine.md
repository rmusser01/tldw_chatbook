---
id: TASK-1463
title: >-
  Enable the 81 never-collected Prompts_DB tests (tests_*.py filename bug) under a quarantine protocol
status: To Do
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - cleanup
priority: medium
dependencies: [task-1452]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Prompts_DB/tests_prompts_db.py` (67 tests) and `tests_prompts_db_properties.py` (14 tests) are named `tests_*` — pytest's `python_files = test_*.py` has never matched them, so ~72KB of that directory's coverage has never executed. Renaming will surface unknown failures; they must be triaged, not silently absorbed. The properties file also carries an import-time `settings.load_profile()` call that task-1452 removes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] Both files renamed to collectable names that do not shadow the existing `test_prompts_db_pytest.py`
- [ ] Directory run triaged: every failure is either fixed, or quarantined `xfail(strict=False, reason="task-NNN: …")` with a filed task, or (if the file proves to be a superseded duplicate of `test_prompts_db_pytest.py`) proposed for deletion with the overlap evidence — owner decides
- [ ] `--collect-only` delta (+81) itemized in the PR
