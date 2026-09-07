---
id: TASK-1463
title: >-
  Enable the 81 never-collected Prompts_DB tests (tests_*.py filename bug) under a quarantine protocol
status: Done
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

- [x] Both files renamed to collectable names that do not shadow the existing `test_prompts_db_pytest.py`
- [x] Directory run triaged: every failure is either fixed, or quarantined `xfail(strict=False, reason="task-NNN: …")` with a filed task, or (if the file proves to be a superseded duplicate of `test_prompts_db_pytest.py`) proposed for deletion with the overlap evidence — owner decides
- [x] `--collect-only` delta (+81) itemized in the PR

## Implementation Plan

1. Rename both `tests_*.py` files to collectable names (`test_prompts_db_legacy.py`, `test_prompts_db_properties.py`)
2. Overlap-check vs `test_prompts_db_pytest.py` before investing in fixes
3. Triage every failure: fix root causes; quarantine only what resists

## Implementation Notes

**All 102 directory tests pass — zero quarantines needed.** Overlap check: only
3 of 67 legacy names collide with `test_prompts_db_pytest.py` (19 tests); the
dormant files are the directory's real coverage, not duplicates. What the
dormancy hid (15 initial failures + 1 deadlock, all root-caused):

- API drift: `add_keyword` returns `Optional[int]` now, not `(id, uuid)` (8
  sites); `add_prompt` requires `author`/`details` (4 sites); the trigger's
  `sqlite3.IntegrityError` propagates raw instead of wrapped `DatabaseError`.
- **Shared-fixture accumulation**: hypothesis runs all examples against one
  function-scoped DB; ten tests assumed per-example freshness (sync-log
  counts, name collisions, soft-delete residue). Restructured with a
  per-example `_fresh_example_db` helper (~6.8ms DDL) + autouse closer.
- **A deadlock that would have killed every full run**: the WAL concurrency
  test's raw UPDATE now trips the version-increment trigger, the writer dies
  before setting its Event, and the reader's UNBOUNDED `wait()` hangs to the
  300s thread-method timeout (the task-1466 shape). Writer now
  trigger-compliant, event set in `finally`, all waits/joins bounded.
- A self-documented bug: `get_prompt_by_id(keyword_id)` with the comment
  "Using wrong get method in original code" — written, noted, never run.
- Unbounded `st.integers()` overflowing SQLite int64 (bounded ±1000); a
  hyphenated uuid4 search term vs `search_prompts`' documented verbatim-MATCH
  contract (hex term; sanitizing belongs to callers per the docstring).

Collection: Tests/Prompts_DB 21 -> 102; full tree 24,505 / 0 errors.
Renamed: both files. Modified: `test_prompts_db_legacy.py`,
`test_prompts_db_properties.py`.
