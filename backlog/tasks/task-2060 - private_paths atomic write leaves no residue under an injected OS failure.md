---
id: TASK-2060
title: private_paths atomic write leaves no residue under an injected OS failure
status: To Do
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
- [ ] #1 `Utils/private_paths`' own test suite injects a failure into `atomic_private_write_bytes`
      (e.g. the write or the rename raising mid-operation) and asserts NO residue remains —
      neither the destination file nor a leftover temp file — proving the `finally` cleanup
- [ ] #2 The test names the exact failure point it injects (write vs rename) so a future change
      to the temp+rename strategy that reopened a residue window would fail it
<!-- AC:END -->
