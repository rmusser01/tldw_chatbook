# Task 5 independent design-gate review

Date: 2026-09-07. Verdict: **STOP / failed approved orderly-shutdown gate**.
This is a design-gate review, not Task 5 implementation or merge approval.

## Artifact and scope

Reviewed `task-5-brief.md`, `task-5-report.md`, and the immutable
`task-5-gate-diagnostic.py` in the original development workspace
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/`.
Both diagnostic snapshot and
untracked test hashed to Git blob `acf4a816ba75fabb1cfe91e405ea0d2ea6db6094`.
Product HEAD was `e8f7ae2ce3c3b6b0b3be69992412d72f7d11d48e`.
Inspected only the existing wrapper/revalidation, repository close/cleanup,
helper cleanup, cooperative lease, and test sandbox code for concrete lifetime
or cleanup confounds. No product changes, full suite, or implementation review.

## Independent evidence

One focused reproduction from this worktree:

```sh
../../.venv/bin/python -m pytest -q Tests/TTS/test_profile_sqlite_helper_lifecycle.py -k 'retained_repository_exit and helper_shape'
```

Result: **1 failed, 1 passed, 3 deselected, 1 warning in 1.66s**, exit 1.
The helper-shaped orderly arm failed only the final preservation assertion:
both substituted names absent, both external observer link counts zero, bytes
still unchanged. The earlier assertions passed: actual helper death/reap and
terminal retained charge, repository quarantine, cooperative EXCLUSIVE refusal
before exit, child return code zero, the stated atexit observation, and EXCLUSIVE
acquisition after exit. The helper-shaped abrupt arm passed preservation and
lock release. The warning was the reported requests dependency compatibility
warning. The two-repository test was deselected and was not rerun.

## Findings and calibration

1. **The stop is supported.** Foreign-name deletion violates the brief even
   though open observer FDs preserve readable bytes. The observer assertions run
   before pytest fixture teardown; the observer owns only foreign sidecars, and
   post-exit lease acquisition accesses the separate `.lock` path. Existing
   `_worker_cleanup` raises on failed exact authority before SQLite close or
   residual-file deletion; `_finish_close` retains ownership on that failure.
   No inspected cleanup path explains the result as fixture restoration/removal.
   The independently reproduced ordinary/abrupt distinction supports stopping
   the approved retention design before implementation.

2. **The helper-shaped arm is a lifetime control, not exact integration
   equivalence.** It uses the real helper and real loss, but the old wrapper
   refuses close because `file_fd == -1`; it does not execute the proposed
   proof-loss exception/latch path. Closing legacy descriptors while SQLite is
   already open also differs from never opening them: it can affect process
   file-lock state, so helper recheck does not establish identical native SQLite
   lock history. This is a remaining equivalence limit, not an observed cause
   invalidating the gate: the reported unmodified baseline also fails, while the
   control removes retained original-file descriptors and evidence SQL as a
   necessary explanation. The result establishes that the tested ordinary
   retained-owner strategy is unsafe; it does not prove every possible future
   helper-wrapper implementation must fail.

3. **Phase attribution must remain narrow.** The observations establish name
   loss after this atexit callback and before completed ordinary child exit.
   They do not prove all atexit callbacks had finished, identify the exact
   native destructor/syscall, or capture its stack. SQLite finalization is the
   supported explanation from retained ownership and the guarded close path,
   not a directly traced C-level fact. `sql_retained`, `lease_retained`, and
   `worker_retained` test non-null owner references; in particular the executor
   reference is not evidence its worker thread remains alive at that callback.
   Actual pre-exit SHARED exclusion is separately established by the contender.

4. **Scope remains bounded.** No complete Textual shutdown, new error mapping,
   admission latch, partial-publication loss, cancellation, restore handoff,
   healthy sibling qualification, or cross-platform shutdown qualification is
   established. No hang occurred in the selected arms. The report correctly
   excludes the initial writer test's masked cleanup failure from post-close
   lock-exclusion evidence.

## Decision

Keep Task 5 stopped pending an explicitly chosen ownership/teardown design
direction. Do not convert the diagnostic to expected failure, weaken foreign
preservation, or infer permission for forced product exit. No additional
implementation is necessary to make this blocker reviewable. The exact native
unlink path and a pristine helper-first lifetime remain unqualified; neither
justifies declaring the current approved orderly-exit gate satisfied.
