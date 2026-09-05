---
id: TASK-31422
title: >-
  Settle the closeout-cycle destination flake's rate disparity with a larger paired sample
status: To Do
assignee: []
created_date: '2026-09-04 00:00'
updated_date: '2026-09-04 00:00'
labels:
  - library
  - testing
  - flaky-test
dependencies: []
references:
  - .superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/task-1-report.md
  - .superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/task-2-report.md
  - backlog/docs/library-decomposition-recipe.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_adaptive_reader_closeout.py::
test_closeout_single_app_route_cycle` is a destination-cycling test (recipe
§16's documented trap: it can look related to an unrelated change without
being one) that has appeared as a branch-unique failure in multiple wave-3
and wave-4 subsystem sweeps. Every investigation so far traces the failure
to the SAME step — `_focus_closeout_work_via_f6`'s "collections has no
reachable Work focus target" assertion, at the 'collections' destination,
processed before the subsystem actually under test in any of these sweeps —
and finds no code-level mechanism connecting it to the diff being verified.

The mechanism has been ruled out repeatedly (most recently in wave-4 Task 1's
own investigation, which re-verified the skills state object's construction
ordering is correct and unrelated to the failing step). What remains
unsettled is the RATE: the same failure signature reproduces on both the
branch and a pristine baseline tree at different OBSERVED rates in small
samples (e.g. wave-4 Task 1's own 11-run sample: 7/8 failed on branch vs.
1/3 failed on baseline, in isolated single-process re-runs), plausibly
explained by this session's heavily fluctuating machine load rather than a
real branch/baseline disparity — but an 11-run sample is not enough to rule
out a real, smaller effect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A paired, quiescent sample of at least 20 isolated single-process runs each of `test_closeout_single_app_route_cycle` — one set against a recent Library-decomposition branch tip, one set against its pristine pre-task baseline, both on an otherwise-idle machine (no other concurrent test/build load) — is captured and recorded.
- [ ] #2 The recorded comparison either confirms the failure rate is statistically indistinguishable between branch and baseline (closing this question for future subsystem sweeps to cite directly, per recipe §7's "documented pre-existing failures" list) or identifies a concrete, reproducible mechanism explaining a real rate disparity.
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: This is a test-flakiness investigation and evidence-gathering task,
not a behavior or contract change.

1. Pick a recent Library-decomposition branch tip (e.g. this wave's own
   final commit) and its pristine pre-task baseline (a `git stash -u` or a
   scratch worktree checkout of the commit immediately before the wave's
   first commit).
2. Confirm the machine is otherwise idle (`ps aux` shows no other long-running
   pytest/build process) before starting either sample, per recipe §19
   lesson 5's own finding that ambient load moves absolute failure counts.
3. Run `test_closeout_single_app_route_cycle` in TRUE isolation (single
   test, single process, `-p no:randomly`) 20+ times against each tree,
   recording pass/fail per run.
4. Compare the two rate distributions; if they diverge meaningfully, dig
   into the 'collections' destination step's own focus-target dispatch
   (`_library_workbench_focus_targets`) for a real, reproducible cause
   before concluding a code-level regression exists.
5. Record the settled conclusion in `backlog/docs/library-decomposition-recipe.md`
   §7's "documented pre-existing failures" list so future subsystem sweeps
   can cite it directly instead of re-investigating from scratch.
