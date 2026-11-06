---
id: TASK-39052
title: "test_closeout_single_app_route_cycle is red on dev: returning to Notes re-arms autosave after the burst's 10 s max wait, so the parked draft the test expects kept is saved at once, and no lane runs the test"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 16:24
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
- [x] #1 The owner decides whether returning to Notes with a parked dirty draft whose max wait has passed saves it at once or waits a fresh debounce; the decision is recorded in this task's notes
- [x] #2 The route-cycle test asserts that decision without racing the autosave timer; it is red before the fix with the n-2 'Closeout route draft' / version 2 diff (or, if the decision keeps today's behaviour, the test's old expectation is rewritten on purpose and named in the notes)
- [x] #3 The test passes 5 of 5 alone and 10 of 10 at `-n 4`, with times and load recorded
- [x] #4 The owner decides whether this roughly 50 s test joins a PR lane; if it does, three timings of that lane's command are recorded
<!-- AC:END -->

## Implementation Plan

1. Record the owner decisions (2026-11-06): save at once on the return (today's behaviour), and lane the test in the Dreams boundary step.
2. Rewrite `_exercise_closeout_single_app_route_cycle` so it cannot race the autosave timer: inside the helper, patch the screen's debounce and max wait so no real timer can fire during the cycle, and drive the burst clock through `library_note_autosave`'s own `time.monotonic()` seam.
3. Pin the rule explicitly: nothing is saved while the draft is parked; the burst clock is then moved past the max wait while the user is away; the return to Notes saves n-2 exactly once ('Closeout route draft', version 1 -> 2); the stored notes and the notes receipt say so. The old "retained without save" expectation is rewritten on purpose.
4. Evidence: the dev test red as filed; the new test red under a temporary source edit that drops the return re-arm; then 5 of 5 alone and 10 of 10 at `-n 4`, with times and load.
5. Lane: append the node to the Dreams boundary step (workflow, `DREAMS_BOUNDARY_TARGETS`, the step argv pin), record three `jobstep.sh` timings, and run `Tests/CI`.

## Implementation Notes

- **Owner decision (2026-11-06):** "Save at once: keep today's behaviour — the max wait means a draft is never left unsaved longer than 10 s of editing time, so the return saves it; fix the route-cycle test to expect the save."
- **Owner decision (2026-11-06):** for the lane, "Dreams boundary step: append it there."
- **AC#1.** Decided by the owner, as quoted. No product code changed.
- **Test (AC#2).** The old expectation (no save across the cycle, `notes_draft_retained_without_save` True) is rewritten on purpose to expect the save. `_exercise_closeout_single_app_route_cycle` no longer races the timer:
  - For the cycle, it patches the screen's `LIBRARY_NOTES_AUTOSAVE_SECONDS` and `LIBRARY_NOTES_AUTOSAVE_MAX_WAIT_SECONDS` to an hour, so no autosave timer can fire on its own. The patches are undone in the helper's `finally`.
  - It drives the burst clock through `library_note_autosave`'s own `time.monotonic()` seam.
  - After the first loop it asserts that nothing was saved while the draft was parked. It then moves the burst clock past the max wait, and asserts that the return to Notes saves n-2 at once and exactly once: `[("n-2", "Closeout route draft", 1)]`, stored as version 2. The debounce it would otherwise wait is the patched hour.
  - The Notes receipt reads the session's real `dirty` (False after the save). The test pins the Notes and Prompt drafts and the new `notes_draft_saved_on_return` observation.
  - This pins an invariant dev already keeps, so its red is shown under temporary source edits, each reverted:
    - dropping the return re-arm (`_select_library_rail_row_after_source_admission`'s `_schedule_library_note_autosave()`) fails with `Returning to Notes did not save the parked draft at once: saves=[]`;
    - having the return wait a fresh debounce (`library_note_autosave_delay` returning the debounce) fails the same way.
  - The old test on base 2684e5453f: red 3 of 3, with the filed n-2 'Closeout route draft' / version 2 diff, when three ran in parallel at load 11-14. Green once alone at load 10 (42.8 s). This is the timer race the task describes.
- **AC#3.**
  - 5 of 5 alone: 38.8, 32.1, 35.7, 39.8 and 29.3 s of pytest time, at load 11.8-14.6 on 18 CPUs.
  - 10 of 10 at `-n 4`, with a temporary 10-repeat parametrize on the test (since removed): 164.1 s, at load 19-40.
- **AC#4 (lane).** The node is appended after the last target of "Run Dreams and Guardian boundary contracts", before `--storage-lease-guard`, in the workflow, in `DREAMS_BOUNDARY_TARGETS` and in the step argv pin in `Tests/CI/test_response_rules_critical_gate.py`.
  - Recent CI: run 42218508176 (#3436, 2026-11-06) took 253 s for this step, and 177 s for the job's admission-sensitive step.
  - Three runs of `bash $FU/jobstep.sh fu-r11-library "Run Dreams and Guardian boundary contracts"` (29 targets, `--storage-lease-guard`, serial). Each gave 871 passed:
    - 510.8 s (8.5 min), at load 24-37;
    - 845.9 s (14.1 min), with the load rising from 24 to 60;
    - 833.8 s (13.9 min), with the load falling from 60 to 31.
  - A fourth run of the same targets with `--durations` took 604.2 s at load 40-66. The new node took 90.3 s of it.
  - The two runs over the brief's 12-minute mark ran at load 40-60 on 18 CPUs. The node itself costs 29-40 s alone, so it is not what took them past 12 minutes. CI ran this step in 253 s.
  - `Tests/CI` at `-n 4`: 1215 passed, 1 skipped.
- **Not changed.** The frozen task-23019 evidence runner (`Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py`) still encodes the old "Notes draft retained without save" receipt. No lane runs it, and it requires its own pinned subject revision. `Tests/Live/test_library_adaptive_reader_closeout.py` names only the node id, which is unchanged.
- **Lesson.** `backlog/docs/lessons-testing-evidence.md`: a `-p` repeat plugin cannot reach a `@private_profile_test` child. The first 10-repeat run here "failed" 10 of 10 in 15 s with `not found` in each child's log.
- Files: `Tests/UI/test_library_adaptive_reader_closeout.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`, `backlog/docs/lessons-testing-evidence.md`.
