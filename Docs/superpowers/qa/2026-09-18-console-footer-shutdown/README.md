# Console footer shutdown guard — TASK-32815

A late setup refresh could rebuild footer hints after Textual popped its last
screen. `App.focused` then raises `ScreenStackError`; the archive flow originally
passed its interaction assertions and failed during teardown. The original
[incident](../2026-09-18-css-consolidation/consumer-red-176.txt) is retained.

Both Console footer focus queries now treat that exact exception as inactive
focus. Ordinary rail/composer focus and hint registration are unchanged. No
style, shortcut, storage or runtime authority changes are involved.

## Verification

- Three real mounted Console regressions deterministically empty Textual's
  live screen stack, verify that `App.focused` raises, exercise each helper and
  the observed `_apply_console_setup_block(False)` path, and restore the stack
  in `finally` without yielding while empty. All three failed before the fix
  with the original exception; [red results](regression-red-results.json) and
  individual failure logs remain here.
- The regressions also verify active rail hints, restoration after the empty
  interval, and their removal when focus moves to the composer.
- [Targeted results](targeted-results.json) record **12 passing cases** in the
  final serial private-profile run: the three regressions, footer registration/recompose, collapsed-composer
  vocabulary, archive recovery and existing environment shutdown guards.
- [Independent review](independent-review.txt) found no actionable issue.
  [Ruff comparison](lint-delta.json) reports no introduced diagnostics; existing
  ChatScreen lint debt remains. Changed method ranges and the complete new
  test file pass Ruff format checks; authored files pass `git diff --check`.

Two additional archive restore/resume/send cases fail during mounting because
`CapturingGateway` lacks `cached_context_window`. Both exact cases also fail on
[saved source 2cc702b85c](archive-baseline-results.json); original and baseline
logs are retained. They are not counted as passing or fixed by this guard.
TASK-32817 tracks the fixture contract and its original send/history assertions.

The runner is the recorded [isolated consumer runner](../2026-09-18-css-consolidation/run_consumer_cases.py).
Cases run serially; no full suite or native graphical run is claimed for this
nonvisual guard. Earlier CSS screenshots retain their original source bound.
The controlled callback reproduces this precise shutdown window, not every
possible late worker or the different setup-blocking-True path.

ADR required: no. ADR-031 governs existing footer truthfulness; TASK-32297 is
the existing empty-screen-stack lifecycle precedent. This is a routine bug fix.
Draft PR2707 remains unmerged and requires its own visual review/approval.
