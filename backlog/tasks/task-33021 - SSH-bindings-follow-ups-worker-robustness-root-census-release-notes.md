---
id: TASK-33021
title: 'SSH bindings follow-ups: worker robustness, root census, release notes'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-27 12:00'
updated_date: '2026-09-27 12:00'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33009
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the deferrals left by TASK-33009 (SSH remote workspace bindings, PR #2838): harden the remote worker against stubbed-import failures, verify the remote denylist against symlinked entries, gate new laptop-disk reads of admitted roots, and record the user-visible fs-tool output change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A workspace folder containing a symlink loop lists successfully (loop skipped), locally and through the remote worker
- [x] #2 No stubbed import in the remote worker bundle raises unhandled outside an explicit laptop-only allowlist, enforced by a test
- [x] #3 A binding rooted at, or above, the real target of a symlinked `~/.ssh` cannot read the keys (pinned by tests)
- [x] #4 A new or increased read of an admitted root's `.root` fails a census test with guidance to handle RemoteRoot
- [x] #5 CHANGELOG records SSH bindings and the fs_* sha256/size result lines
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Symlink loop: `_relative_target_is_safe` caught only `OSError`, but Python <= 3.12 raises `RuntimeError` for a loop in `Path.resolve()`, so one looping entry turned `fs_list` into `worker_failure` -- locally as well as remotely (reproduced both ways). Now catches `(OSError, RuntimeError, ValueError)` and skips the entry.
- `sensitive_paths._debug` imported loguru unguarded; inside the bundle that import is a raising stub, so any resolution-failure diagnostic killed the remote worker. Guarded with `except ImportError`. New bundle test walks every stubbed-import `raise` and requires it to be handled, except the laptop-config denylist builders (reached only without a sensitive context or via git_*, both absent remotely; raising there fails closed).
- Denylist symlink aliasing: the worker already maps each entry through `resolve()` and checks lexical + resolved paths; two loopback tests (root at / above the real target of a symlinked `~/.ssh`) pass on the existing code and now pin it. No code change needed.
- `.root` consumer gate: `Tests/Tools/test_admitted_root_consumer_census.py` pins 26 `file::function` read sites of `authority/selection/admitted.root`.
- Settings `ssh -G` off the UI thread already landed in PR #2838. rc-noise live check still needs a real server (fake-ssh covers the byte-level strip).
- Files: `Tools/local_tool_impls.py`, `Utils/sensitive_paths.py`, `Tools/remote_worker_bundle.py` (regenerated), CHANGELOG, four test files + one new.
<!-- SECTION:NOTES:END -->
