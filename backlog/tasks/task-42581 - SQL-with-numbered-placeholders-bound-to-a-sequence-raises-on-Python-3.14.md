---
id: TASK-42581
title: "SQL with numbered ?1 placeholders bound to a sequence raises on Python 3.14"
status: To Do
assignee: []
created_date: 2026-11-26 19:34
labels:
- follow-up
- db
- python-3.14
priority: medium
references:
- tldw_chatbook/DB/base_db.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-27 Linux lane check (ubuntu:24.04, Python 3.12.3, 2026-11-26), on dev 5dcba48401 and on the round-27 stack alike. Some statement binds numbered placeholders (`?1`) with a sequence of parameters. Python 3.12 warns on every such call: `DeprecationWarning: Binding 1 ('?1') is a named parameter, but you supplied a sequence which requires nameless (qmark) placeholders. Starting with Python 3.14 an sqlite3.ProgrammingError will be raised.` The warning is reported at the generic execute site `tldw_chatbook/DB/base_db.py:729` (`result = run(*args)`), so the statement itself is upstream of it. In the recovered-suites step alone it fired about 468 times, mostly from Console trace tests (`test_console_trace_slot_drop.py` 318, `test_console_route_census_capture.py` 94, `test_console_child_preset_fallback_trace.py` 56). On Python 3.14, which the project plans to move to, every such call would raise instead, so the affected Console trace reads and writes would fail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every statement that binds numbered placeholders to a sequence is found (for example by running the suites with `-W error::DeprecationWarning` scoped to that message, or by a static scan), and the list is in the notes
- [ ] #2 Each is changed to qmark placeholders with a sequence, or to named placeholders with a mapping, with no behaviour change
- [ ] #3 A guard fails when a new statement binds numbered placeholders to a sequence (for example a test that turns that DeprecationWarning into an error for the DB layer)
<!-- AC:END -->
