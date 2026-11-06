---
id: TASK-39052
title: "test_closeout_single_app_route_cycle is red on dev: returning to Notes re-arms autosave after the burst's 10 s max wait, so the parked draft the test expects kept is saved at once, and no lane runs the test"
status: To Do
created_date: 2026-11-06 10:09
labels:
- follow-up
- library
- notes
- tests
- needs-owner-decision
priority: medium
references:
- Tests/UI/test_library_adaptive_reader_closeout.py
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_note_autosave.py
- tldw_chatbook/UI/Library_Modules/screen_constants.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06. Line numbers are from dev. The Library round-10 branch appends tests to this file but leaves this test and its route-cycle helper unchanged, and the test fails the same way there.

**The failure.**
- `Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle` (`:1440`) was red 3 of 3 alone on dev, and 4 of 4 on the Library branch. Each run took 52-60 s wall (load 10-27).
- It fails at `:1399`: `At index 1 diff: {'id': 'n-2', 'title': 'Reading list', 'content': 'Closeout route draft', ..., 'version': 2} != {... 'content': 'bravo', ..., 'version': 1}`.
- No PR lane runs it: no workflow or census entry names the node or the whole file. The census lists only other nodes of this file.

**The cause** (traced with probes on dev):
1. At +9.36 s the test types the draft (`:1209`). `handle_library_note_body_changed` arms the autosave: a 2 s debounce and a 10 s max wait (`screen_constants.py:178`, `:183`).
2. The test switches to Prompts. The note is parked and the timer stops.
3. At +21.63 s the second loop returns to Notes. `_select_library_rail_row_after_source_admission` re-arms the autosave (`library_screen.py:23582-23587`: `retain_note_session and row_id == BROWSE_NOTES and note_snapshot.dirty`).
4. The burst's max wait ran out 12.3 s earlier, so `library_note_autosave_delay` returns the 0.05 s floor (`library_note_autosave.py:107`).
5. At +21.69 s `_fire_library_note_autosave` saves n-2.

- Control: with `LIBRARY_NOTES_AUTOSAVE_MAX_WAIT_SECONDS` patched to 1000, the whole test passed on dev with no save (2 passed in 45.5 s).
- So the red is TASK-34000.1's max wait (e939c9b3cd, 2026-10-04) meeting TASK-34000.25's re-arm on return ("saved only on the return to Notes", `library_note_autosave.py:212-214`).
- The test's `notes_draft_retained_without_save` / `save_calls == []` checks (`:1403`, `:1420`) only held while the test outran the 2 s debounce. They depend on load, which fits TASK-38981's note of one route-cycle flake that was red only on its base.
- Which behaviour is right is a product question: save at once when a parked draft returns past its max wait (what the code does now), or wait a fresh debounce (what the test expects).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides whether returning to Notes with a parked dirty draft whose max wait has passed saves it at once or waits a fresh debounce; the decision is recorded in this task's notes
- [ ] #2 The route-cycle test asserts that decision without racing the autosave timer; it is red before the fix with the n-2 'Closeout route draft' / version 2 diff (or, if the decision keeps today's behaviour, the test's old expectation is rewritten on purpose and named in the notes)
- [ ] #3 The test passes 5 of 5 alone and 10 of 10 at `-n 4`, with times and load recorded
- [ ] #4 The owner decides whether this roughly 50 s test joins a PR lane; if it does, three timings of that lane's command are recorded
<!-- AC:END -->
