---
id: TASK-42581
title: "SQL with numbered ?1 placeholders bound to a sequence raises on Python 3.14"
status: Done
assignee:
- '@claude'
created_date: 2026-11-26 19:34
updated_date: 2026-11-26 20:20
labels:
- follow-up
- db
- python-3.14
priority: medium
references:
- tldw_chatbook/DB/base_db.py
- tldw_chatbook/Chat/console_trace_repository.py
- pyproject.toml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-27 Linux lane check (ubuntu:24.04, Python 3.12.3, 2026-11-26), on dev 5dcba48401 and on the round-27 stack alike. Some statement binds numbered placeholders (`?1`) with a sequence of parameters. Python 3.12 warns on every such call: `DeprecationWarning: Binding 1 ('?1') is a named parameter, but you supplied a sequence which requires nameless (qmark) placeholders. Starting with Python 3.14 an sqlite3.ProgrammingError will be raised.` The warning is reported at the generic execute site `tldw_chatbook/DB/base_db.py:729` (`result = run(*args)`), so the statement itself is upstream of it. In the recovered-suites step alone it fired about 468 times, mostly from Console trace tests (`test_console_trace_slot_drop.py` 318, `test_console_route_census_capture.py` 94, `test_console_child_preset_fallback_trace.py` 56). On Python 3.14, which the project plans to move to, every such call would raise instead, so the affected Console trace reads and writes would fail.

**The statement (round-27 review).** It is `ConsoleTraceRepository.get_effective_owner` (`tldw_chatbook/Chat/console_trace_repository.py`), from d3b996afed (TASK-41310 A). Its recursive CTE names the segment twice as `?1` (`WHERE link.segment_id = ?1), ?1)`) and binds the one-item sequence `(segment_id,)`. Its callers are `ConsoleTraceRepository.reserve_call` and `ConsoleTraceService._validate_owner`.

**Correction (2026-11-26, measured before implementing).** The 3.14 risk above does not hold for this statement. CPython gh-117995 (fixed in 3.12.4 and 3.13) stopped warning when a sequence binds indexed nameless placeholders (`?NNN`). Measured with `SELECT ?1, ?1` bound to `(1,)` under `-W error::DeprecationWarning`:
- Ubuntu 24.04's Python 3.12.3 warns, which is what the Linux check saw;
- local 3.12.11, 3.13.13 and 3.14.7 accept it.

A named placeholder (`:a`) bound to a sequence is the real case: it warns on 3.12 and 3.13 and raises `ProgrammingError` on 3.14.7. The ACs still stand. The statement moves to a named placeholder and a mapping, which silences the 3.12.0-3.12.3 warning. The AC#3 guard matters for named placeholders on every version, and also for `?NNN` on 3.12.0-3.12.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every statement that binds numbered placeholders to a sequence is found (for example by running the suites with `-W error::DeprecationWarning` scoped to that message, or by a static scan), and the list is in the notes
- [x] #2 Each is changed to qmark placeholders with a sequence, or to named placeholders with a mapping, with no behaviour change
- [x] #3 A guard fails when a new statement binds numbered placeholders to a sequence (for example a test that turns that DeprecationWarning into an error for the DB layer)
<!-- AC:END -->

## Implementation Plan

1. Static scan (AST string constants and `.sql` under `tldw_chatbook`) for numbered placeholders. Then a runtime check: the guard below over the recovered-suites step and `Tests/Chat/test_console_trace_*.py`.
2. `get_effective_owner`: `?1` becomes `:segment_id`, bound to `{"segment_id": segment_id}`, with the same line count.
3. A test in the laned `Tests/Chat/test_console_trace_graph_gc.py` calls `get_effective_owner` and `reserve_call` with `warnings.simplefilter("error", DeprecationWarning)`. Red before and green after on Python 3.12.3 (ubuntu:24.04); locally the old statement does not warn, see the Correction above.
4. Guard: a `filterwarnings` entry in `[tool.pytest.ini_options]` that turns `Binding \d+ .* is a named parameter` into an error for every test.

## Implementation Notes

There was one statement. It now binds a named placeholder from a mapping, and a pytest-wide filter fails any test that binds a named placeholder to a sequence.

- **Found (AC#1).** The only statement is `ConsoleTraceRepository.get_effective_owner`: `?1` twice, bound to `(segment_id,)`; from d3b996afed (TASK-41310 A); callers `reserve_call` and `ConsoleTraceService._validate_owner`.
  - Static scan: every SQL-looking string constant in `tldw_chatbook/**/*.py` plus every `.sql` file, for `?NNN`; this is the only hit.
  - Runtime check with the guard below, all with no `is a named parameter` error:
    - locally (3.12.11), the recovered-suites step (2272 passed, 1 skipped) and all 54 `Tests/Chat/test_console_trace_*.py` files (1440 passed, 1 failed);
    - on ubuntu:24.04's Python 3.12.3, after the fix, `test_console_trace_slot_drop.py`, `test_console_route_census_capture.py` and `test_console_child_preset_fallback_trace.py`, the three files behind the Linux check's ~468 warnings: 67 passed, none tripped.
  - The one trace failure, `test_console_trace_current_turn_transforms.py::test_next_send_after_failed_transformed_call[stopped]` (`'error' == 'stopped'`), is red on dev 6e41919880 too, and stays red with the guard overridden.
- **Changed (AC#2).** `?1` became `:segment_id`, bound from `{"segment_id": segment_id}`. The SQL is otherwise the same and the line count is unchanged (size row 4040).
- **Guard (AC#3).** `[tool.pytest.ini_options] filterwarnings = ['error:Binding \d+ .* is a named parameter:DeprecationWarning']`.
  - On every supported Python it fails a test that binds `:name`, `@name` or `$name` to a sequence, the case 3.14 raises on. On 3.12.0-3.12.3 it also fails `?NNN`.
  - `test_a_named_placeholder_bound_to_a_sequence_fails_any_test` pins it: red with the filter overridden on the command line (`DID NOT RAISE`). A scratch probe against the repository config was also red with the entry removed.
  - A global filter was suitable: the step and the trace files pass with it.
- **Red/green.** `test_effective_owner_lookup_binds_its_parameters_by_name` (in the dreams-riders-laned `Tests/Chat/test_console_trace_graph_gc.py`) runs `get_effective_owner` and `reserve_call` under `warnings.simplefilter("error", DeprecationWarning)`. On ubuntu:24.04's Python 3.12.3 with the old statement, it and the file's 5 other `reserve_call` tests fail with the `Binding 1 ('?1')` error; with the fix, 23 pass. Locally it is green both ways, since 3.12.11 does not warn (see the Correction).

Files: tldw_chatbook/Chat/console_trace_repository.py, pyproject.toml, Tests/Chat/test_console_trace_graph_gc.py.

