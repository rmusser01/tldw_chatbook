---
id: TASK-838
title: >-
  path_validation rejects legitimate Windows paths containing a parent-dir segment
status: To Do
assignee: []
created_date: '2026-07-26 23:30'
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
- [ ] `test_lease_constructors_canonicalize_lock_root` passes on windows-latest
- [ ] `validate_path_simple` accepts a legitimate Windows path carrying a parent-dir segment before normalization, and still rejects a genuine traversal outside an intended base
- [ ] Null-byte and shell-expansion rejection is preserved
- [ ] A regression test covers the Windows-style path case on all three platforms
<!-- AC:END -->
