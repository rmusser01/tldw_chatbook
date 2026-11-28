---
id: TASK-42651
title: "A terminated fork child in test_bootstrap_root_cleanup can outlive its 30-second join under load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-28 16:53
updated_date: 2026-11-28 19:39
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
- [x] #4 A SIGTERM or SIGHUP sent to a fork child before its at-fork reset runs still kills it (the signals stay blocked across fork in a process that holds the guard's handler, and are unblocked after the reset)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: a temporary forced-delay edit of the test's fork child (a long big-int multiply, which runs no signal check, between `ready.set()` and `time.sleep`), then a load soak of the unchanged node.
2. Name the race from the reproduction and say which real sessions inherit the handler.
3. Fix it in `Tests/real_profile_guard.py` if the inherited handler is the cause; keep the test's `child=-15 kept=True` assertion.
4. Soak the unfixed and fixed trees at the same load; run `Tests/CI` at `-n 4` and preflight.
5. (AC#4, added 2026-11-28 17:31) Block the guard's signals across fork in `before`, unblock them in `after_in_parent` and after the reset in `after_in_child`; pin a signal sent in the fork window, on both sides, with deterministic tests; re-measure the fork-window probe and rerun the soak and checks.
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
  - The window around fork: real, but not this test's race, because `ready` orders the SIGTERM after the child is running Python. Fixed under AC#4 below.

**Real sessions.** The root conftest installs the handler in a process that owns its root and is not an xdist worker: a serial session or the xdist controller. `Tests/UI/conftest.py` does the same for runs rooted at `Tests/UI`. So in every `-p no:xdist` run (the UI lane, the recovered and admission-sensitive suites, local serial runs), the fork children that tests make inherit it. xdist workers do not install it. Chunking's `ebook_chapters` (`_run_finditer_process`: a fork context, then `terminate()` and `join(1)`) is one such caller. The fix covers every fork child, because it acts where the handler is inherited. It does not cover spawn children between fork and exec (TASK-42470), which run no at-fork hook. The app's own `install_termination_handlers` has the same inheritance and is not changed here (TASK-42653).

**Fix (AC#2).** `remove_sandbox_at_exit` registers `os.register_at_fork(after_in_child=...)`, which sets SIGTERM and SIGHUP back to `SIG_DFL` in a fork child while they are still `remove_then_die`. The kernel then kills the child at once. The owner keeps its handler, and a handler someone else installed is left alone. The test still asserts `child=-15 kept=True`. Its child also prints whether its SIGTERM disposition is the default (`default=True`), which pins the fix deterministically. Red on the unfixed guard: `default=False` (the child still ended -15 in that run); green with the fix (5 passed).

**Soak (AC#3).** The unchanged node, repeated by a scratch parametrize plugin (not committed). Each failure was `child=None kept=True`.
- Unfixed: 3 of 40 failed at `-n 4` (load 6-12), 2 of 200 and 17 of 1000 at `-n 8` (load 9-11), and 8 of 1000 at `-n 16` (load 11-13).
- Fixed: 1000 of 1000 and 3000 of 3000 passed at `-n 8` (load 9-12), and 1000 of 1000 at `-n 16` (load 12-13).
- The final tree, with the new pin: 2000 of 2000 at `-n 8` (load 8-13) and 1000 of 1000 at `-n 16` (load 13-14).

**The window around fork (AC#4, added at the coordinator's request).**
- **Cause.** A SIGTERM sent to a fork child before its at-fork reset ran tripped the inherited C handler and was dropped. CPython clears a fork child's pending Python signals (`_PySignal_AfterFork`) before the at-fork hooks run, and once the reset has run, a flag still set for SIGTERM is skipped because the handler is now `SIG_DFL`.
- **Fix.** The guard's at-fork hooks now hold its signals across the fork:
  - `before` blocks every handled signal whose handler is still `remove_then_die` and that is not already blocked;
  - `after_in_parent` unblocks them, and `after_in_child` unblocks them after the reset.
  - The mask is per thread. The hooks run on the forking thread, which is the child's only thread, and keep what they blocked in a `threading.local`. A handler someone else installed is neither blocked nor reset.
  - A SIGTERM to the parent inside the window stays pending and is delivered at the unblock to `remove_then_die`, which removes the root and dies.
- **Tests.** Three new params, each with a hook registered before the guard's, so it runs inside the window:
  - `fork-child-sigterm-before-its-reset` and `fork-child-sighup-before-its-reset`: the child signals itself, prints `held`, then ends by that signal (`-15`, `-1`) with the root kept.
  - `parent-sigterm-as-it-forks`: the parent signals itself in `before`, prints `held`, then ends -15 with its root removed.
  - Red on the reset-only tree: `'' == 'held'`, `'child=-15 kept=True' == 'held\nchild=-15 kept=True'`, and the same with `-1`. On that tree these signals still killed, because the Python handler ran at once; the red shows they were not held. The drop itself happens before CPython's clear, which no Python hook can reach, so the probe below measures it.
  - Green with the block: 8 passed. 300 repeats of each new param at `-n 8`: 900 passed (load 17-19).
- **Fork-window probe** (`start(); terminate(); join(2)`, a 5 s sleep as the target), 300 runs each, back to back at load 10-12:

  | Guard | Survived |
  |---|---|
  | Unfixed | 78 of 300 |
  | Reset only | 71 of 300 (12 of 300 in an earlier run at load ~6) |
  | Reset plus block | 0 of 300 |
  | No handler | 0 of 300 |

- **Other probes.**
  - `subprocess` with and without `preexec_fn`: the exec'd `sh` still ends -15, so no blocked mask leaks into an exec.
  - The parent's mask is empty after forks.
  - Forks from a non-main thread: 0 of 100 children survived `terminate()` at once, and the thread's mask was restored.
  - A parent forking in a loop and sent SIGTERM from outside: 100 of 100 ended -15 with the root removed.
- **Soak of the original node on the final tree:** 2000 of 2000 at `-n 8` (load 10-15) and 1000 of 1000 at `-n 16` (load 15-18).

**Review fixes (three Minor items).**
- **An exception in the child's reset.** The reset runs in `try`, and the unblock in `finally`. Before this, an exception left the child's SIGTERM and SIGHUP blocked for good, and the mask survives exec. New param `fork-child-unblocks-when-its-reset-raises`: an earlier hook makes `signal.signal` raise inside the reset. Red on 3073335c5c's guard (`True` blocked), green now.
  - Not covered: a `KeyboardInterrupt` from a pending SIGINT that fires at the hook's own entry check, before any of its code runs. Python cannot guard against that, and the docstring says so.
  - A bounded probe (one fork, reaped) with the review's `_thread.interrupt_main` hook fires at that entry line on both trees: the child stays blocked and outlives a SIGTERM for 2 s. When `threading`'s own at-fork hook runs between the two, it takes the interrupt instead, and the child unblocks and ends -15.
- **Nested forks.** Each thread keeps a stack of windows, because a hook can fork again. `before` pushes what it blocked. `after_in_parent` pops the innermost window, or nothing if the stack is empty, and unblocks it. `after_in_child` unblocks the union of the whole stack and clears it, because a child is in every open window at once and no outer `after_in_parent` runs in it.
  - The brief said to pop in both hooks; popping in the child left a child forked inside an outer window blocked.
  - New param `fork-inside-a-fork-window` (a `before` hook that forks once) expects `inner False` and `parent False`.
  - Red on 3073335c5c's single slot: `inner True`, `parent True`. Red on the pop-in-the-child variant: `inner True`. Green now.
- **A handler someone else installed.** New param `fork-child-keeps-a-foreign-sigterm-handler` (`SIG_IGN` set before the guard). It was red under a temporary unconditional reset (`False False`). Before the crash it was also red under a mutation that blocked every handled signal but recorded only the guard's own (`True True`), because the child never unblocked SIGTERM.
  - Review round 2 found a gap: a mutation that blocks and records every handled signal (`ours = set(handled)`) still passed, because the mask was read only after the guard's own unblock. The param now also reads the mask inside the window, from a hook registered before the guard's, and expects `False\nTrue False`.
  - Under that mutation it is red (`True` read inside the window); the previous version of the param passed it. All mutations were reverted.
- **Machine crash, caused by this task (2026-11-28, about 18:16 UTC).** The first draft of `fork-inside-a-fork-window` set its once-flag after `os.fork()`. `os.fork()` runs every `before` hook again, this one included, so the hook recursed, and as the recursion unwound each level forked again. The per-user process table filled (about 10,800 processes) and the machine was rebooted. The flag is now set before the fork, with a comment saying why. Each new fork test was first run under `ulimit -u <current + 60>`.
- **Fork budget** (this replaces an `RLIMIT_NPROC` cap; see below). Every child script starts by wrapping `os.fork` with a per-lineage counter. After `_FORK_BUDGET = 3` calls in one lineage, the wrapper raises `BlockingIOError(EAGAIN, "test fork budget spent")`.
  - A forked child inherits the count, so any runaway tree is bounded at about 2**3 processes, on every platform and in any PID namespace.
  - No param forks more than twice in one lineage, the maximum being `fork-inside-a-fork-window`. Checked: at a budget of 2 all 10 params passed (the file's eleventh test, `--version`, makes no child script); at 1 only that param failed, with the budget error. 3 is just above that maximum.
  - Only `os.fork()` is counted. `multiprocessing`'s fork context calls it at call time, so it is covered; `subprocess` and `posix_spawn` are not, which the comment says.
  - The test's stdout assertion now also shows the child's stderr, so a spent budget is visible whichever assertion fails.
  - **Proof,** under `ulimit -u <current + 120>`, with the once-flag temporarily set after the fork again: the param failed in 0.43 s with `BlockingIOError: [Errno 35] test fork budget spent`. The user's process count stayed at 502-514, against a baseline of about 505. Reverted and checked with `cmp`; nothing was left running.
  - **Runaway probe:** every process, children included, tried 300 forks. The tree stopped at 8 processes (7 forks), all reaped.
  - **Why not RLIMIT_NPROC.** Rounds 1 and 2 of the review had capped each child at `RLIMIT_NPROC = <the uid's counted tasks> + _FORK_HEADROOM` (16, then 64), with an `RLIM_INFINITY` fix and a PID-namespace skip. The coordinator found, in a real `ubuntu:24.04` container, that `NSpid:` shows a single pid (it is relative to the procfs's own namespace), so the skip never triggered. The container cannot be detected that way. For a non-root uid whose other tasks live outside the container, the undercount still turned every fork into a false red. Root is exempt from `RLIMIT_NPROC`, so in a default `docker run` (root, unlimited hard limit) the cap did nothing. Counting from inside a namespace cannot be made reliable, so that cap and its counting were removed.
- **Review round 2 (five Minor items).**
  - Items 1 and 2 (the `RLIM_INFINITY` handling and the PID-namespace skip) were done, then superseded by the fork budget above.
  - Item 3: the foreign-handler param, as above.
  - Item 4: the guard's docstring now says that a `KeyboardInterrupt` at any of the hooks' bytecode checks outside the reset can leave the signals blocked, in the child or on the parent's forking thread. The review's `kiparent.py` shows the parent case: the interrupt fires at `unblock_in_parent`'s entry.
  - Item 5: the test file's comment on the guard was rewritten. It now describes the fork budget.
  - The review's `params2.py` passed all 10 params against the round-2 guard under `ulimit -u <current + 120>`.
- **Authorised serial run** (`-p no:xdist`, so the guard's hooks are installed in the pytest process) of `Tests/Tools/test_remote_session_serve.py`, `Tests/Chunking/test_security.py` and `Tests/Terminal/test_posix_backend.py`:
  - dev 2df861ec7d (a detached scratch worktree, removed afterwards): first run 8 failed, 87 passed, 1 skipped; second run 7 failed, 88 passed, 1 skipped.
  - Tip: 7 failed, 88 passed, 1 skipped in both runs.
  - The 7 are `test_security.py` cases failing with the known local `RecoveryRequired: raw_source_selection_changed`, on both trees.
  - Dev's extra failure in its first run was `test_app_crash_master_close_hangs_up_ordinary_child_but_not_detached_limit` (crash report not written within 8 s, load about 9). Alone it passed 3 of 3 on dev and 3 of 3 on the tip, so it is not a difference between the trees.
- **Soak**, capped by the new process rules: the original node 400 of 400 at `-n 4` on the review-round-1 tree, 0fed76b144 (load 9-10, 816-834 processes). On the final tree, the review's soak of every param in rotation passed 400 of 400 at 4 concurrent (load 7.8, 821-822 processes).

**Checks (final tree, ea2ec487df plus this docs commit).**
- On this Mac:
  - `Tests/CI` at `-n 4`: 2086 passed, 2 skipped (load 7-9).
  - `Tests/CI/test_bootstrap_root_cleanup.py` and `Tests/test_real_profile_guard.py` at `-n 4`: 76 passed, 2 skipped; the review got the same.
  - `./scripts/preflight.sh` passed. `ruff check` and `ruff format --check` passed. `Tests/lint/ruff_gate.py`: RATCHET OK.
- On Linux, run by the coordinator in `ubuntu:24.04` as non-root uid 1001 (Python 3.12.3, hard `RLIMIT_NPROC` unlimited), with ea2ec487df installed through pip:
  - `Tests/CI/test_bootstrap_root_cleanup.py`: 11 passed serially under `ulimit -u 400`.
  - `Tests/test_real_profile_guard.py` at `-n 4`: 4 failed, 61 passed, 2 skipped. The failures are the three `test_an_xdist_workers_session_end_refusal_fails_the_run` params and `test_the_incidents_own_dir_fd_writer_cannot_replace_the_profile_file`.
  - The same 4 fail the same way on dev 2df861ec7d in that container, so this change did not cause them. That file runs in no CI lane, and a follow-up was proposed.

**Files.** `Tests/real_profile_guard.py`, `Tests/CI/test_bootstrap_root_cleanup.py`, `backlog/docs/lessons-testing-evidence.md` (new entry).
<!-- SECTION:NOTES:END -->
