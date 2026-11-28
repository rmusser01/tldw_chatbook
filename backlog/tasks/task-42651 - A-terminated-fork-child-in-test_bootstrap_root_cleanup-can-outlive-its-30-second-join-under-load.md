---
id: TASK-42651
title: "A terminated fork child in test_bootstrap_root_cleanup can outlive its 30-second join under load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-28 16:53
updated_date: 2026-11-28 17:26
labels:
- follow-up
- flake
- tests
priority: medium
references:
- Tests/CI/test_bootstrap_root_cleanup.py
- Tests/real_profile_guard.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen once on 2026-11-27, in round 28's final gate (macOS, 18 CPUs), in the fast PR contract step at load 7-26: `Tests/CI/test_bootstrap_root_cleanup.py::test_owned_root_is_removed[fork-child-terminated]` failed with `assert 'child=None kept=True' == 'child=-15 kept=True'` after 30.05 s. The child script starts a `multiprocessing` fork child, waits for it to signal ready, calls `terminate()` and `join(30)`; the child was still alive after the join. The node passed in the same gate's whole `Tests/CI` run, and 3 of 3 times alone, both on that tip and on dev.

The fork child inherits `remove_sandbox_at_exit`'s SIGTERM handler (`Tests/real_profile_guard.py`), which every pytest session installs: it removes the root only in the owning process, resets the disposition and re-sends the signal to itself. Why the child did not die within 30 s is not known. Round 24 saw a similar macOS loss of SIGTERM around fork and exec (TASK-42470).

The handler runs in every test session, and the test's own comment notes that Chunking's `ebook_chapters` strategy terminates fork children the same way, so this may not be limited to the test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The race is named with evidence (a forced-delay or load reproduction), and the notes say whether the inherited handler affects real test sessions or only this test
- [x] #2 It is fixed where it lives, and the test still asserts that a terminated fork child ends by SIGTERM while the parent's root is kept
- [x] #3 A soak under load passes where the unfixed version fails at a measured rate
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: a temporary forced-delay edit of the test's fork child (a long big-int multiply, which runs no signal check, between `ready.set()` and `time.sleep`), then a load soak of the unchanged node.
2. Name the race from the reproduction and say which real sessions inherit the handler.
3. Fix it in `Tests/real_profile_guard.py` if the inherited handler is the cause; keep the test's `child=-15 kept=True` assertion.
4. Soak the unfixed and fixed trees at the same load; run `Tests/CI` at `-n 4` and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The race (AC#1).** The fork child inherits `remove_sandbox_at_exit`'s `remove_then_die` as a Python-level SIGTERM handler. CPython runs a Python handler only at its next bytecode check (or when a blocking call returns EINTR); the C handler only sets a flag. `time.sleep` does not check before it enters `nanosleep`, so a SIGTERM that lands after the child's last check and before the syscall is used up by the C handler, and the 60 s sleep runs to term, past the 30 s join. When the parent exits, multiprocessing terminates the daemon child again. That SIGTERM interrupts the sleep, so the run still exits 0 after about 30 s, which is the gate's signature (`child=None kept=True`, 30.05 s).
- Survivor probe: the test's child script, with a 5 s sleep and a 3 s join, 8 at a time. 1 of 48 survived. The SIGTERM was sent at 0.003 s. `ps` showed no pending or blocked signal. `sample` put its one thread in `time_sleep` → `nanosleep` → `__semwait_signal`. The guard's handler ran at 5.004 s, when the sleep ended, and the child then ended -15.
- Forced delay (temporary test edit, reverted): a 41 ms stretch with no check, eight `x in blob` scans of a 256 MiB `bytes`, between `ready.set()` and the sleep. On the unfixed guard 10 of 10 runs went red with `child=None kept=True` after 30.07 s; on the fixed guard 10 of 10 were green. A big-int multiply and a `hashlib` call did not reproduce it: longobject's `SIGCHECK` and the check after a call run the handler.
- CI: no failing log of the node was found in the 7 latest failed Derived Artifacts runs (searched read-only on 2026-11-28); the evidence is local.
- Other candidates:
  - `ready.set()` returning early: ruled out. `Event.wait` cannot return before the child has released the condition's lock.
  - The handler's `remove()`: ruled out. It is a no-op in a child.
  - The window around fork: real, but not this test's race, because `ready` orders the SIGTERM after the child is running Python. With `start(); terminate()` at once, 79 of 300 children survived the unfixed guard, 12 of 300 the fixed one, and 0 of 300 with no handler. Listed as a follow-up.

**Real sessions.** The root conftest installs the handler in a process that owns its root and is not an xdist worker: a serial session or the xdist controller. `Tests/UI/conftest.py` does the same for runs rooted at `Tests/UI`. So in every `-p no:xdist` run (the UI lane, the recovered and admission-sensitive suites, local serial runs), the fork children that tests make inherit it. xdist workers do not install it. Chunking's `ebook_chapters` (`_run_finditer_process`: a fork context, then `terminate()` and `join(1)`) is one such caller. The fix covers every fork child, because it acts where the handler is inherited. It does not cover spawn children between fork and exec (TASK-42470), which run no at-fork hook. The app's own `install_termination_handlers` has the same inheritance and is not changed here (follow-up).

**Fix (AC#2).** `remove_sandbox_at_exit` registers `os.register_at_fork(after_in_child=...)`, which sets SIGTERM and SIGHUP back to `SIG_DFL` in a fork child while they are still `remove_then_die`. The kernel then kills the child at once. The owner keeps its handler, and a handler someone else installed is left alone. The test still asserts `child=-15 kept=True`. Its child also prints whether its SIGTERM disposition is the default (`default=True`), which pins the fix deterministically. Red on the unfixed guard: `default=False` (the child still ended -15 in that run); green with the fix (5 passed).

**Soak (AC#3).** The unchanged node, repeated by a scratch parametrize plugin (not committed). Each failure was `child=None kept=True`.
- Unfixed: 3 of 40 failed at `-n 4` (load 6-12), 2 of 200 and 17 of 1000 at `-n 8` (load 9-11), and 8 of 1000 at `-n 16` (load 11-13).
- Fixed: 1000 of 1000 and 3000 of 3000 passed at `-n 8` (load 9-12), and 1000 of 1000 at `-n 16` (load 12-13).
- The final tree, with the new pin: 2000 of 2000 at `-n 8` (load 8-13) and 1000 of 1000 at `-n 16` (load 13-14).

**Files.** `Tests/real_profile_guard.py`, `Tests/CI/test_bootstrap_root_cleanup.py`, `backlog/docs/lessons-testing-evidence.md` (new entry).
<!-- SECTION:NOTES:END -->
