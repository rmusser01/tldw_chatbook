---
id: TASK-41701
title: "A deeply nested v2 trace bundle crashes Import with an uncaught RecursionError"
status: Done
assignee:
- '@claude'
updated_date: 2026-11-18 04:45
created_date: 2026-11-18 04:00
labels:
- follow-up
- trace
- security
priority: high
references:
- tldw_chatbook/Chat/trajectory_import.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
dependencies:
- TASK-41604
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A v2 bundle nested about 1400 or more levels deep makes `load_imported_trace` raise `RecursionError`. It escapes the `except TrajectoryImportError` in `TrajectoryScreen.action_open_trace`, so Import crashes instead of reporting the file. Found by the TASK-41603 implementer and confirmed by review r16tr-rev1, on dev 45b1949ca3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A deeply nested v2 bundle fails Import with a named `TrajectoryImportError` and no crash
- [x] #2 A test feeds a bundle nested past the recursion limit (deep in a part validation actually reaches) and is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce on the branch: a resigned bundle with a payload nested past the recursion limit raises `RecursionError` from `_validate_v2` (digest, masking).
2. Catch `RecursionError` around validation in `load_imported_trace` and raise a named `TrajectoryImportError`, as `_read_document` reports a document `json` cannot parse and as the v1 path's `_json_object` already catches it; cover the v1 validator too, whose error quotes the file's `format`.
3. Test in `Tests/Chat/test_trace_v2_collaboration.py`, red first; lane it with the round's other Trace nodes.

## Implementation Notes

A file nested past what validation can walk now fails Import with a named error.

- **Reproduced** on the branch (probe through a file path, as the screen loads): v2 bundles nested 1400 and 3000 levels in an event payload raised an uncaught `RecursionError` (900 imported). `json` parses them (`_read_document`); `_validate_v2`'s masking (TASK-41133's re-mask on import) and digest walk them.
- **Fix.** `load_imported_trace` catches `RecursionError` around v2 validation and around the v1 path, and raises `TrajectoryImportError("'<file>' is not a valid trajectory trace: nested too deeply")` (`_nested_too_deeply`), the wording `_read_document` uses when `json` itself refuses the nesting, so the screen's `except TrajectoryImportError` shows it (with `markup=False`, TASK-41604). The v1 branch needed it too: the v1 validator's error quotes the file's `format` (`_masked`, and `repr` before it) and a deep `format` raised the same way. The v1 mapping step already reported a deep payload as a mapping error.
- **Test.** `Tests/Chat/test_trace_v2_collaboration.py::test_a_deeply_nested_file_fails_import_with_a_named_error`: a resigned v2 bundle whose first event payload is nested `sys.getrecursionlimit() + 500` levels (built through JSON text, so no recursive helper touches it), and a v1 file whose `format` is nested as deep. Red before the fix: 2 failed with `RecursionError: maximum recursion depth exceeded`; green after. The probe now gives the named error at 1400 and 3000.
- **Lane.** The node joins the recovered-suites step with the round's other Trace nodes (5.4 s for the three nodes' 17 cases serially under `--storage-lease-guard`, load 29-31).
- Files: `tldw_chatbook/Chat/trajectory_import.py`, `Tests/Chat/test_trace_v2_collaboration.py`.

**Review r16tr-rev2 (fix round 2), Minor 4.** A deep v1 unknown field or tool payload still read `... could not be mapped ... (RecursionError ...)`: `_v1_snapshot`'s mapping boundary caught it as `Exception`. It now re-raises `RecursionError`, so `load_imported_trace` names it "nested too deeply" like the other shapes. The test gains `v1-unknown-field` and `v1-tool-payload`. Both failed with 7f03daa295's import (the mapping message) and pass now. (`_json_object` is now `_parsed_json`, TASK-41603.)
