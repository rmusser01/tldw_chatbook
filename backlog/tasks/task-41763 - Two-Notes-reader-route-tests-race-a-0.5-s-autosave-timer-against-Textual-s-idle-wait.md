---
id: TASK-41763
title: "Two Notes reader route tests race a 0.5 s autosave timer against Textual's idle wait"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
- flaky
priority: low
references:
- Tests/UI/test_library_notes_reader.py
- tldw_chatbook/UI/Library_Modules/library_note_autosave.py
- TASK-32298
- TASK-32456
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two tests in `Tests/UI/test_library_notes_reader.py` share a race:
- `test_reader_route_parks_dirty_note_selection_and_preview_without_saving` (`:1147`);
- `test_reader_route_invalidates_autosave_queued_before_park` (`:1232`).

Each patches `LIBRARY_NOTES_AUTOSAVE_SECONDS` to 0.5 (`:1200`, `:1281`), presses Notes, and asserts that the route's end re-armed the autosave timer (`:1210`, `:1296`). About 0.2 s of process CPU follows the arm, and `pilot.pause()` waits for the process to go idle. If the process gets less than about 40% of a core in that window, the timer fires first and clears `autosave_timer`, so the assert reads `None`. The arrangement has been on dev since 4f327279dc (2026-08-27).

**A different race on the same two nodes.** TASK-32298 (In Progress) owns `test_reader_route_invalidates_autosave_queued_before_park` for a whole-file failure: `KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"`, raised when the Notes editor's `TextArea` repaints during its teardown at the Notes -> Media switch. Its product guard, a batched Library canvas teardown, is TASK-32456. That is a style-teardown race, not this timer race. TASK-32298's notes also saw this task's assertion fail: wrapping the teardown in `app.batch_update()` shifted the timing and broke `test_reader_route_parks_dirty_note_selection_and_preview_without_saving` at `assert screen._notes_state.autosave_timer is not None`.

**Evidence.**
- **Root-cause investigation, 2026-11-17** (for the media series, on the chain over dev 5e3c8b2bd4):
  - paired concurrent runs at load 38-70 passed 1 of 6 on each tree, and all 6 pairs agreed;
  - a CPU co-tenant thread made both trees red, 2 of 2.
- **Re-check, 2026-11-18,** at dev 908e9840aa under the private-profile marker at `-n 3`: both tests were red in 2 of 2 rounds (load 16-49), each with `autosave_timer` `None`.

Locally the file stops at `RecoveryRequired` without the marker, and no PR lane runs it. That is how, for an afternoon, the failure was mistaken for a slice 3.2 regression.

**A second, latent race.** The timer armed on typing is really 10 s, not 3600: the burst max-wait caps it (`library_note_autosave.py:107`). A Media round trip longer than 10 s therefore saves before the park, and `save_calls == []` fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests pass 5 of 5 with a CPU co-tenant thread running from the Notes press to the assert
- [ ] #2 Over 10 runs of each at -n 4 under load (under the private-profile marker if they still stop at RecoveryRequired locally), none fails at an autosave_timer assertion, with the load recorded; a text-area--gutter KeyError counts against TASK-32298, not this task
- [ ] #3 Their assertions still check the park, the preview, the re-armed timer and that nothing saved before the park
<!-- AC:END -->
