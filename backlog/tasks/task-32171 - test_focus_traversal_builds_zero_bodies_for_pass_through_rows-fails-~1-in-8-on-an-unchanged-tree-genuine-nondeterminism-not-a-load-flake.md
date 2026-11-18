---
id: TASK-32171
title: >-
  test_focus_traversal_builds_zero_bodies_for_pass_through_rows fails ~1-in-8 on
  an unchanged tree (genuine nondeterminism, not a load flake)
status: Done
assignee:
  - '[rmusser01]'
created_date: '2026-09-09 10:03'
labels:
  - library
  - tests
  - flaky-test
  - phase-c
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_library_media_reader_traversal_t22207.py::test_focus_traversal_builds_zero_bodies_for_pass_through_rows fails about 1 run in 8 in isolation on an unchanged tree. Confirmed production-revert-controlled during phase-C task 3: 'git checkout <base> -- tldw_chatbook/' then running the test alone STILL failed, which no production-caused regression can survive; eight runs per arm gave 7 passed / 1 failed on BOTH the branch and the base commit -- an identical rate. Two prior phase-C reports labelled it a 'load flake, passes in isolation', which was too generous: it fails alone, at a stable ~1-in-8 rate. It keeps costing attribution time in every paired sweep of the media battery, because a genuine intermittent is indistinguishable from a branch-caused regression until the revert control is run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The source of the nondeterminism in the zero-body pass-through traversal path is identified (systematic-debugging: find the varying input, likely a settle/ordering/timing-dependent assertion)
- [x] #2 The test is made deterministic -- passes N-in-N or fails 0-in-N reliably over at least eight isolated runs -- OR the production nondeterminism it legitimately catches is found and fixed
- [x] #3 The fix is verified with n>=8 isolated runs on the current tree showing a stable rate, recorded in the task's implementation notes
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: Test-determinism investigation and fix; no production contract change.

1. Verify the premise at the current base: the 2026-09-09 observation predates
   the ADR-126 config-participant admission (TASK-32628), which now fails the
   whole Library battery closed before the test body runs. If confirmed,
   unmask via the sanctioned `keep_bootstrap_profile` enrollment
   (TASK-32873 precedent) so the body can execute at all.
2. Reproduce statistically: >=8 isolated single-process runs of the named
   node, recording pass/fail and the failing assertion each time.
3. Find the varying input (settle-timer firing mid-traversal vs keystroke
   pacing, row ordering, worker scheduling); instrument rather than guess.
4. Fix the TEST to be deterministic (or the production race if real), with
   red->green run counts (N before, N after).

## Implementation Notes

ADR required: no

ADR path: N/A

Reason: test determinism plus a bounded one-shot-guard release that mirrors
two existing patterns in the same file; no new contract or storage
decision.

### Premise verification at dev tip 2612fc56b2 (2026-10-02)

The 2026-09-09 observation predates the ADR-126 config-participant
admission (TASK-32628): standalone, the whole traversal file now dies at
`_flow_app -> _build_test_app -> load_settings` with
`RecoveryRequired: raw_source_selection_changed` (the TASK-33370 class).
Unmasked via `keep_bootstrap_profile` enrollment (the sanctioned
TASK-32873 treatment), the file runs and the flake is still live:

- Reproduction BEFORE any fix (single node, isolated processes):
  20-run loop: 19 pass / 1 fail; 30-run loop: 29 pass / 1 fail —
  ~2/50 with the captured signature
  `Failed: Detail call for backing id 23 did not start` (the 2 s
  `_wait_for_detail_call` deadline expiring). Raising that deadline to
  the suite's 15 s contention convention did NOT close it: a further
  40-run loop still failed 2 with the same "did not start" — the final
  row's fetch genuinely never dispatches in those runs.
- A sibling node in the same file,
  `test_loading_banner_paints_in_place_without_body_rebuild`, flakes the
  same family (~1-in-10 standalone): `assert 0 == 1` on the final
  body-build count.

### Root cause (proven by instrumentation, not inferred)

Class-level tracing of `on_descendant_focus` on a probe copy of the test
(3x15 parallel loops; every failing run showed the same state):

- All 11 `DescendantFocus` events (row 0 -> row 10) ARE delivered and none
  are stale (`event.widget is screen.focused` for every one).
- `_select_library_media_reader_row` is NEVER called: in failing runs
  `screen._library_notes_restoring_focus` is stuck `True` from before the
  traversal begins, so every focus event classifies as `programmatic` and
  the selection block in `on_descendant_focus` is skipped for every row.
  Session state at failure: still `selected=loaded=media:28` (row 0),
  `detail_calls=[28]`, zero selections during the whole traversal.

The leak: `_library_notes_restoring_focus` is a one-shot guard for canvas
recompose intervals ("the queued callback clears it before yielding
through paint"), but the callback lives in `_sync_library_canvas(then=...)`
chains that a suppressed/superseded sync never runs, and — unlike
`_library_notes_resize_settling`, which
`_mark_library_notes_user_interaction` clears on every real key/mouse
event — nothing on the input side ever releases it. A user arrowing the
Items list after an unlucky canvas sync gets a dead list: the Reader never
opens on focus. That is a REAL product defect the test legitimately
catches, and the scheduling variance of the sync-vs-callback race is the
"varying input" AC#1 asked for.

### What shipped

Production (the real fix):

1. `tldw_chatbook/UI/Screens/library_screen.py` —
   `_sync_library_media_browse_state` now captures `_sync_library_canvas`'s
   return and releases `_library_notes_restoring_focus` +
   `_library_notes_programmatic_focus_target` when the sync was
   refused/suppressed, mirroring the media-trash path twelve lines below,
   which already did exactly this ("Projection-suppressed and failed
   targeted syncs do not run the queued callback").
2. `tldw_chatbook/UI/Library_Modules/library_notes_controller.py` —
   `_mark_library_notes_user_interaction` (called from `on_key` and
   `on_mouse_down`) now also clears `_library_notes_restoring_focus`: a
   real input event ends the restore-interval suppression exactly as it
   already ended resize suppression. The programmatic TARGET is
   deliberately kept so the system's own single restore focus stays
   excluded from selection semantics via `target_restore`.

Tests (determinism belt-and-braces):

3. `Tests/UI/test_library_media_reader_flow.py` —
   `_wait_for_detail_call` deadline 2 s -> 15 s (wall-clock contention
   survival, the `_wait_for_condition` convention; only elongates the
   failure path).
4. `Tests/UI/test_library_media_reader_traversal_t22207.py`:
   - `_load_row` waits for the 2.0 s list-entry-focus arm
     (`LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS`) to discharge before any
     caller traverses (the armed receipt likewise classifies focus as
     programmatic);
   - the three probes with final `counts["body"] == 1` assertions
     (pass-through, banner, 1 MB) now WAIT for the deferred document
     projection to land (`counts["body"] >= 1`) instead of closing the
     counting window on a predicate-true early return or a fixed pause
     count — the settle projects through
     `call_next(_recompose_library_media_detail_if_unrendered)`, which
     checked-first polling can observe one tick BEFORE it runs (the
     sibling `0 == 1` manifestation).
5. `Tests/conftest.py` — `test_library_media_reader_traversal_t22207.py`
   enrolled in `keep_bootstrap_profile` (admission mask; the seam-level
   fix for the whole battery is TASK-33370's).

### Evidence (commands + run counts)

Single node, isolated single-process runs, worktree `.venv`
(Python 3.12.13, pytest 9.1.1), `-p no:randomly`:

- BEFORE fixes: 2/50 fail (suppression signature) + sibling 0==1 flake
  (~1/10 on a 15-run loop, 3 failures observed).
- AFTER test-side fixes only (deadline/arm/window): 3/60 fail under
  3-way-parallel load — still the suppression signature (product leak
  unfixed), proving the test-side changes alone were insufficient.
- AFTER the two product fixes: 0 suppression failures in 120
  3-way-parallel-load runs (57/3 then 56/4; every remaining failure was
  `HelperTimeoutError: private_sqlite_helper_timeout` raised in
  `private_sqlite_process.start` during app boot under the deliberate
  3-concurrent-worker storm — a different, load-infrastructure class
  owned by the private-sqlite area, not this test's logic).
- Serial isolated runs after the full fix: 12/12 passed (8-13 s each;
  /private/tmp/t32171serial, 2026-10-02).
- Whole file after the full fix: `6 passed` (51 s) — repeated across the
  session; default pytest-randomly order also 6 passed once the fixes
  landed.
