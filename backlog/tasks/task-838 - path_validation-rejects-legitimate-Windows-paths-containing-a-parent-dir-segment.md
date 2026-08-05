---
id: TASK-838
title: >-
  path_validation rejects legitimate Windows paths containing a parent-dir
  segment
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 23:30'
updated_date: '2026-07-27 13:50'
labels:
  - security
  - windows
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surfaced while confirming whether a red CI check on PR #941 was pre-existing. It was -- and the underlying cause is a real cross-platform bug.

`Tests/Model_Artifacts/test_operation_leases.py::test_lease_constructors_canonicalize_lock_root` fails on **windows-latest only**, at `tldw_chatbook/Utils/path_validation.py:327`, with a ValueError reporting a dangerous pattern for the Windows parent-directory segment.

`validate_path_simple` scans the **raw** path string against a `dangerous_patterns` list that includes the Windows parent-directory form. On Windows that substring appears in ordinary, legitimate paths before canonicalization -- which is exactly what a *canonicalize* function is handed. Because the check runs on the un-normalized string, it cannot distinguish a traversal attempt from a path that merely needs resolving.

This matters more now than it did. PR #941 wired `validate_path_simple` into the Evals snippet importer to satisfy CLAUDE.md's security requirement. That call site wraps it in `try/except ValueError` and degrades to a notification rather than crashing, so it is not a live defect there -- but every new consumer inherits the same Windows over-rejection, and the rule actively tells contributors to add consumers.

The fix is to resolve or normalize before pattern-matching, so a parent-directory segment is judged after canonicalization against an intended base rather than as a substring, and to reserve the raw substring scan for genuinely unresolvable inputs such as null bytes and shell-expansion markers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_lease_constructors_canonicalize_lock_root` passes on windows-latest
- [x] #2 `validate_path_simple` accepts a legitimate Windows path carrying a parent-dir segment before normalization, and still rejects a genuine traversal outside an intended base
- [x] #3 Null-byte and shell-expansion rejection is preserved
- [x] #4 A regression test covers the Windows-style path case on all three platforms
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify diagnosis (raw-substring asymmetry between POSIX '../..' and Windows '..\\').
2. Fix dangerous_patterns Windows entry to require two consecutive parent refs ('..\\..\\'), matching POSIX parity. Do not add normalize-first (would loosen POSIX semantics for this base-less, many-caller security helper).
3. Add regression tests in Tests/Utils/test_security_enhancements.py exercising the Windows-style string directly on POSIX, plus a still-rejected multi-level case.
4. Revert-check: confirm new test fails on old code, passes on fixed code.
5. Run targeted test files that exercise path_validation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the raw-substring dangerous_patterns list in validate_path_simple() so the
Windows parent-ref entry requires two consecutive parent refs ("..\..\"),
matching the existing POSIX entry ("../.."). Previously the Windows entry
("..\") matched a single parent ref, so the same logical path (e.g.
"nested/../locks") was accepted on POSIX and rejected on Windows -- this is
what broke test_lease_constructors_canonicalize_lock_root on windows-latest,
since ArtifactOperationLease hands validate_path_simple a path that still
needs .resolve() to canonicalize.

Took the minimal parity fix, not the normalize-first approach the task
description floated. Evaluated normalize-first: os.path.normpath would
collapse an inner "a/b/../../c" segment (no base needed for that), but that
changes current POSIX behaviour -- a path like "a/b/../../c" is REJECTED
today (raw string contains "../..") and would become ACCEPTED after
normalizing first ("a/c"). The task explicitly warned against loosening
POSIX behaviour, and this function is deliberately base-less (containment is
validate_path()/validate_path_safety()'s job), with ~25 call sites across
the codebase. A one-line parity fix closes the exact asymmetry with zero
POSIX behaviour change and a much smaller risk surface, so I did not
implement normalize-first.

Left the "~/" / "~\\" entries untouched -- confirmed via grep that
Local_Ingestion/local_file_ingestion.py already relies on the current
"expanduser-before-validate" contract, and the task flagged another
in-flight PR depends on it too. Not part of this bug.

Added two tests to Tests/Utils/test_security_enhancements.py::TestValidatePathSimple:
- test_single_parent_ref_accepted_both_separator_conventions: asserts
  "/tmp/xyz/nested/../locks" and "C:\Temp\xyz\nested\..\locks" are BOTH
  accepted -- exercises the Windows-style string directly so it's
  meaningful on POSIX CI, not gated behind a platform skip.
- test_multi_level_parent_ref_still_rejected_both_conventions: asserts
  "../../etc/passwd" and "..\..\etc\passwd" are BOTH still rejected, so the
  fix doesn't weaken the check.

Revert-checked by hand (git stash is off-limits in this shared worktree):
reverted the source change via Edit, ran the two new tests --
test_single_parent_ref_accepted_both_separator_conventions failed with
"ValueError: Path contains dangerous pattern: ..\" exactly as expected;
test_multi_level_parent_ref_still_rejected_both_conventions passed (it
should, old code already rejected via the shorter "..\" substring). Restored
the fix via Edit and reran -- both pass, diff matches the original edit
exactly.

Testing: ran every test file under Tests/Utils/ plus every file found via
`grep -rl path_validation Tests/` plus
Tests/Model_Artifacts/test_operation_leases.py (the originally failing
test). 851 passed, 0 failed, 20 pre-existing warnings (unrelated
RuntimeWarning/PytestUnraisableExceptionWarning noise from
Tests/UI/test_mcp_workbench.py's MCPWorkbench._clear_tool_view coroutine,
not touched by this change). Did not run the full suite (out of scope per
instructions; ~18k tests).

Modified files: tldw_chatbook/Utils/path_validation.py,
Tests/Utils/test_security_enhancements.py.
<!-- SECTION:NOTES:END -->
