---
id: TASK-41130
title: "Seed the Library Collections reader layout from the measured width so the first compose does not close the Library pane"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-16 21:24'
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- focus
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_collections_state.py
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** On a first visit the Collections reader shell composes with a width-0 layout, which closes and disables the Library pane, so a focus restore aimed at the Collections row finds it unfocusable. TASK-40328 review r14ls-rev2 fixed the symptom with a one-time retry; seeding the layout from a real width removes the cause.

**Evidence.** `library_collections_state.py:156-162`: `reader_layout` defaults to `resolve_adaptive_reader_layout(0, AdaptiveReaderLayoutPreferences(), LIBRARY_COLLECTIONS_READER_PROFILE)`. `library_collections_controller.py:270` (the retry): "a first visit composes this shell with the width-0 …" layout; the shell's first real layout sync reopens the pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The first Collections compose uses a layout resolved from the screen's measured width, so the Library pane is open and focusable in the first frame
- [x] #2 The focus retry is removed if it is no longer needed, and the Enter-on-Collections-row test still passes
- [x] #3 A test checks the first frame's pane state, red before the change
<!-- AC:END -->

## Implementation Plan

1. Confirm at base: a probe records the layout the Collections shell is composed with and the screen width at that compose, for a rail press and for a Library mounted with Collections already selected. Base: both compose `(library_open=False, items_open=False, widths 0)` while `screen.size.width` is already 170.
2. Red test: hold the shell's layout sync back and press Enter on the Collections row; the composed shell's first layout must have the Library pane open and the row focused (red at base: the width-0 layout).
3. Seed: when the stored layout is still the width-0 placeholder, resolve it from `screen.size.width` (same preferences, profile and `reader_has_item` as the shell's sync) before composing the shell, and store it.
4. Remove the r14ls-rev2/rev3 focus retry (`_library_collections_focus_retry` and its bookkeeping) if the seeded layout makes it dead; rewrite its pin (`test_collections_focus_retry_never_moves_a_users_own_focus`, a census node) into the first-frame test, renamed in place in the census.
5. The Enter-on-Collections-row test (`test_rail_press_recomposes_library_once_and_loads_captures_in_place`) must still pass; run the covering files, ratchets, ruff, preflight.

## Implementation Notes

- **Premise held at base cc6006fcf5** (2026-11-15). A probe recorded the layout the Collections reader shell is constructed with: `(library_open=False, items_open=False, widths 0)`, both for a rail press and for a Library mounted with Collections already selected. `screen.size.width` was 170 at that compose in both cases, so the width was already known.
- **Fix.** `LibraryCollectionsController._seed_library_collections_reader_layout` runs from the Collections branch of `compose_content`. While the stored layout is still the width-0 placeholder (all three widths 0) and the screen has a width, it resolves the layout from `screen.size.width` and stores it. `_resolve_library_collections_reader_layout` is now the one resolver for the seed and the shell's layout sync (same preferences, profile and `reader_has_item`). The first layout sync still refines the result against the measured shell (4 columns narrower above 120 columns, where the shell grid has its border and padding). Its `previous` is now the seed. Hysteresis only keeps a closed pane closed, and the seed is never more closed than the shell's own resolve, so the first sync gives the same result as before. At 124-127 columns the screen width opens the Library pane and the shell's does not, so the first frame shows the pane and the sync closes it, moving focus to its grip. The retry used to leave focus on that grip too.
- **Retry removed** (AC#2): the r14ls-rev2/rev3 retry in `_sync_library_collections_reader_layout_from_shell`, `LibraryScreen._library_collections_focus_retry` and its bookkeeping in `restore_focus_after_recompose`. Negative control with the seed reverted and the retry gone: `test_rail_press_recomposes_library_once_and_loads_captures_in_place` (Enter on the Collections row) fails in both parameters with focus on `library-ingest-top-button`; with the seed it passes.
- **Test** (AC#3): `test_collections_first_frame_opens_the_library_pane[rail_press|restored]` holds the shell's layout sync back, so only the first frame is checked. The shell's layout has the Library pane open, the pane is displayed and enabled, and for the rail press Enter leaves focus on the row both before and after the sync is released. Red before the seed in both cases (`AdaptiveReaderEffectiveLayout(library_open=False, ..., widths 0)`), green after. It replaces `test_collections_focus_retry_never_moves_a_users_own_focus`, whose subject is gone; its census entry is renamed in place (count unchanged). Serial with the lane's flags: 32.0 s call / 51.9 s total at load 82-92, and 13.3 s call at load ~46.
- **A test that read the first frame as "measured".** `test_library_crit9_shell.py::test_escape_returns_to_the_library_pane_below_64_columns[browse-collections]` failed 3 of 4 on the seeded head and passed 4 of 4 at base. Its `_narrow_stage_layout` wait took non-zero widths to mean the layout sync had run, so it now pressed Escape before the shell was measured, and the pane toggle's sync returned early. Product behaviour in that window is unchanged: an Escape in the first frame was lost the same way at base. The wait now also requires `screen._library_adaptive_reader_allocation_is_current(shell)`, the gate the sync itself checks: 4 of 4, and the whole crit9 file passes 25 of 25. Lesson added to `lessons-testing-evidence.md`.
- **Covering runs**: see TASK-41126's notes (the same runs cover both commits); `test_library_shell.py`'s restore and focus subset and the crit9 file are listed there.
- **Size.** `library_collections_controller.py` 1733 -> 1740 (measured; pin raised with a dated comment). `library_screen.py` 38188 -> 38180; its screen-ratchet ceiling is left at 38188 because the parallel Library streams edit that row.
- Files: `tldw_chatbook/UI/Library_Modules/library_collections_controller.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_collection_reader_journeys.py`, `Tests/UI/test_library_crit9_shell.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `backlog/docs/lessons-testing-evidence.md`.
