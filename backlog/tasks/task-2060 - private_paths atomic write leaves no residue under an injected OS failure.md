---
id: TASK-2060
title: private_paths atomic write leaves no residue under an injected OS failure
status: In Progress
assignee: []
created_date: '2026-08-02'
labels:
  - testing
  - private-paths
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from task-1719 (phase-2b audio coverage residuals). AC#2 there added a test proving the
briefings audio pipeline leaves no orphan DESTINATION file when `atomic_private_write_bytes`
raises — which holds by construction, because `atomic_private_write_bytes`
(`Utils/private_paths.py`) writes to a temp file and only renames onto the destination after full
success, with a `finally` that unlinks the temp on any exit.

That temp-file-cleanup guarantee is `private_paths`' own responsibility, and its own test suite
has NO test that injects a raise into the real temp+rename path to prove the temp file is
actually cleaned up (no residue) under a synthetic OS failure mid-write. The guarantee is
currently asserted by code inspection only, not by a test. This is a real, separate coverage gap
surfaced (not hidden) during task-1719 — in a different module, so out of that task's scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Utils/private_paths`' own test suite injects a failure into `atomic_private_write_bytes`
      (e.g. the write or the rename raising mid-operation) and asserts NO residue remains —
      neither the destination file nor a leftover temp file — proving the `finally` cleanup
- [x] #2 The test names the exact failure point it injects (write vs rename) so a future change
      to the temp+rename strategy that reopened a residue window would fail it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two tests in `Tests/Utils/test_private_paths.py`, each naming its injection point (AC#2):
`test_no_residue_when_the_temp_write_side_fails` (injects at the TEMP-FILE fsync — payload fully
written, pre-rename) and `test_no_residue_when_the_rename_itself_fails` (injects at the rename of
the temp onto the destination, selectively: only calls whose args end in `.tmp`). Both assert the
destination retains its ORIGINAL content and that the directory's contents are IDENTICAL
before/after (a snapshot comparison — strict against any residue shape, immune to unrelated
tmp_path entries). Mutation-verified: disabling the `finally` temp-unlink in
`atomic_private_write_bytes` reds BOTH.

Trap found and closed en route: `_atomic_posix_guards_available` checks
`{os.rename, os.unlink} <= os.supports_dir_fd` by FUNCTION IDENTITY, so monkeypatching `os.rename`
made the first version of the rename test pass VACUOUSLY (early bail with
`required_posix_guards_unavailable`, no temp ever created — every assertion green for the wrong
reason). The mutation protocol caught it (the test stayed green under the disabled-cleanup
mutation). Fixed by pinning the guards check True for that test's duration, documented in its
docstring; the write-side test exercises the unpatched check on the same run.

Test-only change; production `private_paths.py` untouched.
<!-- SECTION:NOTES:END -->
