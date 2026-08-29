---
id: TASK-23151
title: Library on_resize does unconditional Notes work on every frame again
status: To Do
assignee: []
created_date: '2026-08-28'

status: Done
assignee: []
created_date: '2026-08-28'
updated_date: '2026-08-28 23:32'
labels:
  - performance
  - library
  - regression
priority: high
dependencies: []

dependencies: []
priority: high
---

## Description



<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-08-02 ratchet `test_library_note_fifty_same_side_resize_sequences_do_zero_notes_work`
asserts that a resize which does not cross a layout band does **zero** Notes work. It now measures
**300** calls to `_apply_library_notes_stage_visibility` across 50 resizes (both parametrised
initial sizes). This is a genuine production regression, not a stale test: the ratchet is
correct and must stay at `== 0`.

This is the same defect class TASK-23025 eliminated from the resize path days earlier — per-frame
work reaching the DOM on frames that changed nothing — so it also erodes a fix the 2026-08-27
performance review just shipped.

## Acceptance Criteria

- [ ] A same-side resize sequence that crosses no layout band performs zero Notes stage-visibility
  work, with the existing ratchet unchanged and still asserting `== 0`
- [ ] Band-crossing resizes still apply stage visibility exactly once per crossing
- [ ] The emergency-return path added by the introducing commit keeps its behaviour (a regression

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A same-side resize sequence that crosses no layout band performs zero Notes stage-visibility
  work, with the existing ratchet unchanged and still asserting `== 0`
- [x] #2 Band-crossing resizes still apply stage visibility exactly once per crossing
- [x] #3 The emergency-return path added by the introducing commit keeps its behaviour (a regression
  test covers the narrow-emergency case it was added for)

## Evidence

`tldw_chatbook/UI/Screens/library_screen.py:7284` calls `_apply_library_notes_stage_visibility()`
**before** the `if compact == self._library_notes_compact: ... return` early-out at `:7285-7287`
that makes same-side resizes a no-op. 50 resizes x 6 call sites = 300. The same commit added the
call to `_update_library_notes_responsive_state` (`:7237`).

Introduced by `6161bd1fe19` (2026-08-26) "feat(library): add narrow emergency return path", on dev
via merge `6bed8d6f59` (PR #2124). Reproduces standalone, so it is not test pollution.


<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read `6161bd1fe19` and establish WHY the call sits above the crossing return before moving
   anything; measure the ratchet's real call counts first.
2. Decide between the two shapes the evidence allows -- relocate the call below the crossing
   early-out, or gate it on a changed stage signature (the TASK-23025 pattern in this file).
3. Implement the chosen shape and prove the other one is wrong with a mutation.
4. Prove a band CROSSING still applies stage visibility exactly once, for both bands.
5. Run the whole of `Tests/UI/test_library_shell.py` plus the Library resize/reader/honesty files
   and compare failure name sets against pristine dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Shape chosen: gate, not move.** Reading `6161bd1fe19` shows the call was added above the
crossing early-out because the ordinary emergency band (`LIBRARY_EMERGENCY_WIDTH`, 64 cells) is a
DIFFERENT band from `LIBRARY_NOTES_COMPACT_BREAKPOINT` (120). A 63 <-> 64 resize crosses only the
former, and the commit's own tests drive exactly that (`...restore_is_defeated_by_newer_return_
interaction` resizes 80 -> 63 -> 64 -> 63 -> 64, never touching 120). Moving the call below the
crossing return therefore strands the emergency takeover entirely -- verified by mutation: the new
regression test fails with "63-column resize never engaged the emergency stage".

The fix adds `_library_notes_stage_signature()` -- a cheap tuple (cached refs from
`_library_layout_ref`, plus flags; no DOM walk) of exactly the inputs
`_apply_library_notes_stage_visibility` turns into writes -- and
`_apply_library_notes_stage_visibility_for_resize()`, which the two resize legs
(`on_resize` and `_update_library_notes_responsive_state`) now call in place of the raw leg. The
signature carries the EFFECTIVE emergency decision (`_library_ordinary_route_active()` AND the
width bucket), not the raw width bucket, because on the non-ordinary routes the geometry is inert
and the raw bucket flips for nothing.

Two things the first attempt got wrong, both found by measurement rather than reasoning:

- Carrying `rail.display`/`canvas.display` unconditionally kept the wide case at 100 calls. Under
  an adaptive reader shell those are owned by the reader's own `sync_layout`, which hides the rail
  purely as a function of width -- and the stage leg returns before its own toggles in that branch,
  so it never reads them. They are now carried only while the legacy path owns them.
- Recording the applied signature only inside the resize gate left exactly 1 call per burst (the
  first frame after any other seam had applied). `_apply_library_notes_stage_visibility` is now a
  thin wrapper that clears the record, runs the legs (`_apply_library_notes_stage_legs`, the
  unchanged body), and records the POST-apply signature -- so every one of the screen's ~20 seams
  arms the gate, and a leg that raises leaves it re-armed rather than stale.

Measured with the ratchet itself: 201 and 100 applications before, 0 and 0 after. Crossings still
cost exactly one application each -- compact band both ways, twice over (pinned in
`test_resize_compact_crossing_still_transitions_both_ways_twice`), and the emergency band in and
out (`test_stage_visibility_runs_once_per_emergency_band_crossing`, which is born-red against BOTH
wrong shapes: 2 != 0 against the unconditional call, and a stranded emergency stage against the
relocated one).

**Verification.** `Tests/UI/test_library_shell.py` runs 823 passed / 0 failed with
`-p no:randomly` on this branch, so there is no failure name set to diff -- it is empty, a strict
subset of any dev baseline (which carries the two ratchet params plus the TASK-23153 in-file
pollution). `Tests/UI/test_library_resize_focus_gates_t23025.py` 9 passed;
`test_library_notes_reader.py` + `test_library_entry_compose_once.py` +
`test_library_honesty_accessibility.py` 139 passed. `./scripts/preflight.sh` green.

**Review round (Qodo, Testability).** The gate was covered only through the Textual harness, so
its own branches had no focused test. Added 8 unit tests driving the shipped functions -- taken
unbound from `LibraryScreen` onto a fake screen (`_StageGateScreen`), with only the leg itself
replaced by a counter -- covering skip (unchanged signature), apply (changed signature, exactly
once, then re-armed), fail-open (unavailable signature, where both sides of the comparison are
`None`), and the three shapes measurement forced: the effective-not-raw emergency decision, the
rail/canvas display carried only while the legacy path owns them, and the record being taken
inside `_apply_library_notes_stage_visibility` so every seam arms the gate (plus the clear-first
ordering that keeps a raising leg from leaving a stale record). Each assertion was mutation-tested
against the branch it covers; the four single-line shape mutations each redden exactly one test.
`Tests/UI/test_library_resize_focus_gates_t23025.py` now 17 passed, both ratchet params still 0.

Modified files: `tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_resize_focus_gates_t23025.py`,
`backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
