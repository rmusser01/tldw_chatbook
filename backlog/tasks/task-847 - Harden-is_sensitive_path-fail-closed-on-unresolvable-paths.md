---
id: TASK-847
title: Harden is_sensitive_path fail-closed on unresolvable paths
status: Done
assignee: []
created_date: '2026-07-27 02:36'
updated_date: '2026-07-27 05:06'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
is_sensitive_path's docstring promises it fails CLOSED for a path it cannot resolve, but _resolved() catches only OSError and RuntimeError. A path containing a NUL byte raises ValueError instead of returning True. Unreachable today because validate_path rejects such a path first and the file tools carry an outer catch-all, but the comment overstates the guarantee. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 is_sensitive_path returns True (refused) for a path that cannot be resolved for any reason including ValueError,A regression test covers the NUL-byte case,The docstring's fail-closed claim matches the code
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce first: confirm is_sensitive_path raises ValueError (uncaught) for a NUL-byte path instead of returning True, proving the fail-closed docstring promise does not hold for every resolution failure.
2. Broaden _resolved()'s except clause from (OSError, RuntimeError) to Exception, matching the lazy-accessor pattern already used elsewhere in the module (log at debug, return None).
3. Add a real (unmocked) regression test exercising the NUL-byte path end to end, alongside the existing monkeypatched fail-closed test.
4. Re-verify the docstring's fail-closed claim now matches the implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced first: `is_sensitive_path(Path("bad\x00path"))` raised `ValueError: lstat: embedded null character in path` instead of returning True -- `_resolved()`'s except clause only caught `(OSError, RuntimeError)`. Broadened it to catch any `Exception` (matching the lazy-accessor pattern already used elsewhere in the module: debug-log then return None), so ANY resolution failure now returns None and is_sensitive_path returns True. `_resolved()` got a new docstring explaining why the broad catch is required for the fail-closed guarantee to actually hold; `is_sensitive_path`'s own docstring already stated the guarantee correctly (it was aspirational until now), so no separate docstring rewrite was needed there.

Added a real (unmocked) regression test, `test_nul_byte_path_fails_closed_for_real`, alongside the pre-existing monkeypatched `test_unresolvable_path_fails_closed`.

Verified: `pytest Tests/Utils/ Tests/Tools/ Tests/Agents/ -q` -> 893 passed, 0 failed.

Files: tldw_chatbook/Utils/sensitive_paths.py, Tests/Utils/test_sensitive_paths.py.
<!-- SECTION:NOTES:END -->
