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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two tests in `Tests/UI/test_library_notes_reader.py` share a race:
- `test_reader_route_parks_dirty_note_selection_and_preview_without_saving` (`:1147`);
- `test_reader_route_invalidates_autosave_queued_before_park` (`:1232`).

Each patches `LIBRARY_NOTES_AUTOSAVE_SECONDS` to 0.5 (`:1200`, `:1281`), presses Notes, and asserts that the route's end re-armed the autosave timer (`:1210`, `:1296`). About 0.2 s of process CPU follows the arm, and `pilot.pause()` waits for the process to go idle. If the process gets less than about 40% of a core in that window, the timer fires first and clears `autosave_timer`, so the assert reads `None`. The arrangement has been on dev since 4f327279dc (2026-08-27).

**Evidence.**
- **Root-cause investigation, 2026-11-17** (for the media series, on the chain over dev 5e3c8b2bd4):
  - paired concurrent runs at load 38-70 passed 1 of 6 on each tree, and all 6 pairs agreed;
  - a CPU co-tenant thread made both trees red, 2 of 2.
- **Re-check, 2026-11-18,** at dev 908e9840aa under the private-profile marker at `-n 3`: both tests were red in 2 of 2 rounds (load 16-49), each with `autosave_timer` `None`.

Locally the file stops at `RecoveryRequired` without the marker, and no PR lane runs it. That is how the failure passed for a slice 3.2 regression for an afternoon.

**A second, latent race.** The timer armed on typing is really 10 s, not 3600: the burst max-wait caps it (`library_note_autosave.py:107`). A Media round trip longer than 10 s therefore saves before the park, and `save_calls == []` fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests pass 5 of 5 with a CPU co-tenant thread running from the Notes press to the assert
- [ ] #2 Both pass 10 of 10 at -n 4 under load, with the load recorded
- [ ] #3 Their assertions still check the park, the preview, the re-armed timer and that nothing saved before the park
<!-- AC:END -->
