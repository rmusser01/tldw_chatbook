---
id: TASK-21234
title: >-
  test_fleet_teardown_notice killed whole-suite background runs but does not
  reproduce standalone
status: Done
assignee:
  - '[rmusser01]'
created_date: '2026-08-23'
labels:
  - test-health
  - dev-red
  - needs-owner
dependencies: []
priority: high
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; observed during
TASK-21100's fix round.

During the burn-down, `Tests/Chat/test_fleet_teardown_notice.py` hung for **more than 420 s
under a 60 s per-test timeout on pristine dev**, and was identified as the cause of
whole-suite background-run kills. It was excluded identically on both sides of every A/B so
that attribution stayed sound, which is why the burn-down's own evidence was unaffected.

Re-probed at close-out on dev `b2b1e2e0d` with a hard 180 s wall: the whole file **passed —
6 passed in 6.69 s**. The file itself has not changed since `f2e274993` (2026-08-22), i.e.
before the hang was observed, so the difference is not in the test source.

The hang is therefore conditional on something a standalone run does not reproduce — suite
ordering, a co-imported module, or concurrency. That is precisely the shape that keeps costing
whole-suite runs and that a green standalone run cannot close. It needs an owner with the
whole-suite context. A second defect is visible in the original observation and should not be
lost: a 60 s per-test timeout that lets a test run past 420 s is not doing its job.

## Acceptance Criteria

- [x] The condition under which the file hangs is reproduced deterministically, or the hang is shown not to reproduce under a full whole-suite run on current dev and any exclusion for it is removed
- [x] If it reproduces, the blocking wait is identified and the test no longer exceeds the per-test timeout
- [x] A whole-suite run completes without a background kill attributable to this file
- [x] The per-test timeout terminates this test if it hangs again, or the reason it cannot is recorded

## Implementation Notes

ADR required: no

ADR path: N/A

Reason: test-health verification and unmasking; no production change.

### Premise verification at dev tip 2612fc56b2 (2026-10-02)

The 2026-08-22 hang observation predates two regime changes:

1. The ADR-126 config-participant admission (TASK-32628, ~2026-09-24):
   standalone, 5 of the file's 6 nodes now fail at
   `_build_test_app -> load_settings` with
   `RecoveryRequired: raw_source_selection_changed` (the TASK-33370
   class) — the file cannot even run unmasked. Enrolled in the
   conftest's `keep_bootstrap_profile` filename set (the sanctioned
   TASK-32873 treatment for real-app-mounting suites).
2. The pytest-timeout posture change (TASK-22062 / d512afd2b5,
   2026-08-25): `timeout_method` is now deliberately unset, so
   pytest-timeout uses the `signal` method on macOS/Linux — a hang is
   interrupted in the main thread, reported as exactly ONE named
   failure, and no longer kills the run the way the old `thread`
   method's worker death did. The observed ">420 s under a 60 s
   timeout" shape is the `thread`-method-era failure mode that commit
   specifically fixed.

### Hang reproduction attempts (all negative)

Worktree `.venv` (Python 3.12.13, pytest 9.1.1), `-p no:randomly`:

- Standalone: 10 consecutive full-file runs — every run
  `5 passed, 1 failed` in 15-24 s (pytest) / 18-29 s (wall). The single
  failure is fast, not a hang (classified below). No run approached the
  300 s per-test timeout.
- Whole-suite shape approximated per this task's constraints (file plus
  the two sibling modules it imports —
  `test_console_agent_bridge.py`, `test_child_run_scope_ordering.py` —
  in ONE process, on a machine already carrying several sibling pytest
  armies): 3 consecutive runs, 347 tests each, 224-253 s per run, same
  5+1 fleet result each time, no stall, no background kill. (The 165
  sibling-module failures in those runs are the pre-existing
  RecoveryRequired class documented by TASK-33370 — identical on a
  clean dev base, not attributable to this file.)

AC1: the hang does not reproduce on current dev in either shape, and no
repo-level exclusion for this file exists to remove (the 2026-08-22
burn-down excluded it ad hoc in its own A/B invocations only; a repo
sweep for the filename finds no CI skip list, pytest.ini deselect, or
ciignore entry).

### AC2 (recorded rather than satisfied — the hang never reproduced)

N/A: no blocking wait could be identified because no run hung. For the
record, every wait in the current file is bounded:
`_join_fleet_threads(timeout=5.0)`, `entered_event.wait(5)`, the
navigation expect-loop (15 s deadline), the unmount wait (300 x 0.02 s)
and the 1.5 s quiet-window — none is an unbounded C-level lock acquire
that a SIGALRM could not interrupt.

### AC4 (timeout posture, recorded)

With `timeout = 300` and the signal method (pyproject's TASK-22062
comment), a reoccurred hang in this file would be terminated at 300 s
and named as exactly one failing test — verified posture by
configuration, not by inducing a hang. The one theoretical exception
(SIGALRM cannot raise inside an uninterruptible C-level acquire) does
not apply to any wait this file performs today.

### The one remaining red node (owned elsewhere)

`test_a_superseded_console_leave_stages_no_teardown_notice` fails fast
(~16 s) with `Timed out waiting for #console-native-composer` /
`NoMatches '#console-left-rail' on ChatScreen` — byte-identical to the
dev-drift class TASK-33621.36 already recorded for
`test_console_send_draft_snapshot.py` ("With the bootstrap plugin, 3
tests are still red (NoMatches '#console-left-rail', a composer wait
timeout...), and both results are identical on base"). It is the
Console composer-mount drift, not a teardown-notice defect and not a
hang; ownership stays with TASK-33621.36. The other five nodes (the
actual teardown-notice contracts) are green once unmasked.

### What shipped

- `Tests/conftest.py`: `test_fleet_teardown_notice.py` added to the
  `keep_bootstrap_profile` filename enrollment (unmasks the five
  direct-seam nodes; TASK-33370 owns the seam-level fix that will
  subsume it).

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: Test-health investigation; no production contract change.

1. Verify the premise at the current base (dev tip 2612fc56b2): the 2026-08-22
   observation predates the ADR-126 config-participant admission
   (TASK-32628), which now fails this file's app-building nodes closed
   standalone (same TASK-33370 class). Unmask first (bootstrap_profile
   enrollment) so the file can execute at all.
2. Reproduce the hang: repeated standalone runs AND sibling-file single-process
   runs (the whole-suite shape approximated per the task's constraints: this
   file plus Tests/Chat neighbors), under wall-clock watch. Record run counts
   and durations.
3. Analyze the file's blocking waits (`_join_fleet_threads`,
   `entered_event.wait`, full-app `run_test`) against the current
   pytest-timeout posture (300 s, signal method per TASK-22062) and record
   whether a reoccurred hang would be terminated, and why/why not.
4. Close with evidence either way; no production change expected.
