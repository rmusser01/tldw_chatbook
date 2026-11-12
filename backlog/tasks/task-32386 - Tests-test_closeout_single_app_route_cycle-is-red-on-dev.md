---
id: TASK-32386
title: 'Tests: test_closeout_single_app_route_cycle is red on dev'
status: Done
assignee:
  - '[rmusser01]'
created_date: '2026-09-11 10:30'
labels:
  - tests
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle` fails on `origin/dev` itself, not on any wave branch -- it was A/B'd against a clean `git archive` of the base during the critique-10 wave (Tasks 6 and 7 both reported the identical failing name on base). The contract entry the test asserts is `.library-media-route` (`Tests/UI/test_library_adaptive_reader_closeout.py:59`) while the screen reports `library-browse-reader-shell` -- `assert 'library-browse-reader-shell' == '.library-media-route'`. See also task-31422, which is open against the same test node id (its flake-rate disparity), so the two should be settled together. A permanently red test on the default branch trains everyone to read a red file as noise, which is how a real regression gets waved through.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Tests/UI/test_library_adaptive_reader_closeout.py` passes on origin/dev
- [x] #2 The resolution states whether the screen or the assertion was wrong, rather than deleting the check
- [x] #3 If the assertion was stale, it now names the selector the screen actually mounts
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: Test-only flake/dev-red resolution; no production contract change.

1. Verify the premise at the current base (origin/dev tip 2612fc56b2): run the
   named node standalone and in its file, and record the actual failure
   signature versus the one the task filed (assertion vs admission).
2. If the config-participant admission (ADR-126, TASK-32628/32873 class) masks
   the body before the assertion can run, apply the sanctioned enrollment
   (`keep_bootstrap_profile` filename set in `Tests/conftest.py`) so the test
   executes, per TASK-32873's precedent.
3. With the test executing, triage the original assertion
   (`receipt["identities"]["shell"]` vs `DESTINATION_CONTRACT[media][1]`):
   read the production media reader shell's mounted id/classes and decide
   whether the screen or the assertion is right; fix the stale side.
4. Re-run the whole file N times (red->green evidence with run counts).

## Implementation Notes

ADR required: no

ADR path: N/A

Reason: test-flake/dev-red resolution at the test seam plus one production
comment-only context change; no storage/sync/provider/contract decision.

### Premise verification at dev tip 2612fc56b2 (2026-10-02)

- Standalone, the whole file (14 nodes) failed at app construction with
  `RecoveryRequired: raw_source_selection_changed` (ADR-126 config-participant
  admission; the TASK-32628/32873/33370 class) — a mask that postdates this
  task's filing (2026-09-11) and hides the filed assertion entirely.
- Unmasking (bootstrap-profile file enrollment as an experiment) restored
  execution and reproduced the filed failure EXACTLY:
  `assert 'library-browse-reader-shell' == '.library-media-route'` at the
  receipts identity assertion. Classification: (c) known-class mask +
  genuine stale assertion.

### Root cause (the filed failure)

`7b4a7e2896` (2026-09-08, phase-C "one resident browse shell") replaced the
per-route shell ids with ONE `LibraryBrowseReaderShell` whose id is
deliberately route-neutral (`LIBRARY_BROWSE_READER_SHELL_ID =
"library-browse-reader-shell"`; naming it after either route "would lie")
and whose route is projected via the marker classes
(`.library-media-route` / `.library-notes-route`). The closeout contract's
media and notes entries were updated to the marker-class selectors so
queries kept working, but the receipts assertion
`shell.id == contract_selector.removeprefix("#")` still assumed an ID
selector — stale for exactly those two destinations (the loop failed at
media first). **The screen was right; the assertion was stale.**

### What shipped

1. `Tests/UI/test_library_adaptive_reader_closeout.py`
   - Every case now runs under `@private_profile_test` (fresh profile per
     case): passes the config-participant admission AND keeps each case's
     durable per-destination reader preferences isolated (a shared
     in-process bootstrap profile leaks the cycle case's Notes-Items-close
     into the later Notes cases' initial layout — observed as
     `items.region.width == 0` when file-level enrollment was tried).
   - The identity assertion now pins `LIBRARY_BROWSE_READER_SHELL_ID` for
     the two shared-shell destinations and keeps the id-selector
     comparison for the other four (check kept, not deleted; the constant
     is imported from production so there is one source of truth).
   - `_focus_closeout_work_via_f6` now WAITS OUT the destination mode-swap
     recompose before asserting the Work focus target: the mode press
     flips `reader_mode` before the work-pane swap lands, and a cold/slow
     event loop reached the assertion mid-recompose (work pane unmounted,
     target list empty) — the exact `collections has no reachable Work
     focus target` signature TASK-31422 chased as a rate disparity.
     Diagnostic state (row, work pane id/display/mounted, available pane
     ids, focus) is in the failure message.
   - The Notes branch-paging case waits for real Items-pane geometry
     before asserting containment (region lags DOM mount; observed
     transient `assert 0 > 0` under load).
   - The conversations leg waits the row list back after the stale-A read
     starts (observed transient `assert 0 >= 2` mid-repaint).
2. No production change was required for the filed assertion.

### Evidence (commands + run counts)

Worktree `.venv` (Python 3.12.13, pytest 9.1.1), `-p no:randomly`:

- Before: `pytest Tests/UI/test_library_adaptive_reader_closeout.py` →
  14 failed (admission mask); single node → failed 3/3 at admission.
- After enrollment experiment (mask only): whole file 5 failed / 9 passed,
  with the filed identity assertion reproduced on the cycle node.
- After the full fix: whole file `14 passed` — 3 consecutive runs
  (185.40 s, 205.94 s, 214.95 s). Single node
  `test_closeout_single_app_route_cycle`: 1 passed — runs at 44-65 s
  (private-profile child); 6-run loop after the final conversations-leg
  wait: 6/6 passed (see /private/tmp/t32386-ev2 logs, recorded 2026-10-02).
- Neighbors unaffected: `Tests/UI/test_library_media_reader_traversal_t22207.py`
  6 passed (separate task's enrollment), `Tests/UI/test_app_quit_guard.py`
  20 passed.

### Owner follow-ups (not this task)

- The remaining pre-existing reds referenced here are owned elsewhere:
  TASK-33370 (the admission seam for the ~135-file RecoveryRequired class,
  including the rest of the Library battery), TASK-31422 (the closeout
  focus-target rate disparity — materially advanced by the settle-wait
  here, which converts the race into a bounded wait), TASK-33621.36
  (Console composer-mount drift: `#console-native-composer` /
  `#console-left-rail` — same signatures seen on the fleet-teardown
  suite's full-app node under TASK-21234).
