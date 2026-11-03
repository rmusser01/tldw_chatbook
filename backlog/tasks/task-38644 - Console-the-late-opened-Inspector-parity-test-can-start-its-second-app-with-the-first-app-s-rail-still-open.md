---
id: TASK-38644
title: "Console: the late-opened Inspector parity test can start its second app with the first app's rail still open"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-02 21:52
updated_date: 2026-11-03 01:14
labels:
- follow-up
- tests
- console
- ci-flake
priority: medium
references:
- Tests/UI/test_console_collapsed_inspector_send_layout.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-37000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02). `Tests/UI/test_console_collapsed_inspector_send_layout.py::test_an_inspector_opened_after_a_send_matches_one_open_throughout` is a UI census entry, so a failure fails the required check.

**CI failures.** It failed twice in UI Fast Lane (4) on 2026-10-31, both times at `assert not rail.display` (:302), which means the second app started with the Inspector open:
- run 41189972193 attempt 1, on fu/h-settingsreds (#3315), whose only product change was `settings_screen.py`; green on rerun;
- run 41224394415 attempt 1, on fu/h-railstate @ edd6260383 (#3225's own branch), which changed Console rail-state code (`chat_screen.py`, `Chat/console_rail_state.py`), including d02046bf61 "retain first-use rail defaults across restarts".

**Candidate causes.** The two candidates below are not yet told apart.
- **Worker-shutdown race.** The test boots two apps in one private profile, so they share `config.toml`. The first app collapses the rail, then relies on a 0.3 s pause (:312-315, "Leave the stored rail preference collapsed for the next arm"). That save is a background-thread worker (`_save_console_rail_preferences`, chat_screen.py:11815), and Textual's shutdown cancels workers without waiting for them. So the second app can read the rail as still stored open. This fits the #3315 failure, which ran without any rail-state change.
- **Retained first-use seeds.** d02046bf61's change reached dev as 71ddbecdbb in #3225 (merged 2026-10-31 22:24Z). Since then, a mount with no stored rail record latches a first-use seed, which is restored from the sidebar UI-state file. `_console_rail_state_config()` (chat_screen.py:11808) re-injects that seed only when the selected key is absent; an explicit config record wins.
  - So an open seed wins only in three cases: the first app stored its collapse under a different scope key from the second app's selected key; the save did not land, which overlaps the race above; or a delayed seed write overwrites the record.
  - The last case is the stale-seed write that `test_console_rail_writes_cannot_finish_with_stale_seed` pins, and that test is red on dev (TASK-38643, cause C).

**A second symptom, locally.** In 12 runs under `-n 4` load at dev 88af9f9379, it failed once at `assert opened_late == open_throughout` (:352). The rail opened after the send painted no "▼ more sections — scroll" hint. That is TASK-37000.5's missing fold hint, here without a chat switch.

TASK-37000 (Done) built this test, which last changed in a9ea81a09f (2026-10-27).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the open-rail start is identified (worker-shutdown race, retained first-use seed, or another) and recorded; if it is a product rule, that rule is pinned rather than worked around
- [x] #2 The second app starts with the Inspector collapsed in 10 of 10 runs under -n 4 load, and not because of a longer fixed pause
- [x] #3 The witness-row, fold-hint and painted-frame comparisons stay as strict as they are now
- [x] #4 Any remaining fold-hint mismatch is recorded on TASK-37000.5 with this test's evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Instrument with a scratch pytest plugin (loaded through `PYTEST_PLUGINS`, since the private-profile child drops `-p` and `PYTEST_ADDOPTS`): record each rail save's dispatch, start and end, and at each Console mount the retained seeds, the in-memory and on-disk rail records, and the state of the earlier app's save workers. Run under `-n 4` load, naturally and with the collapse save slowed.
2. Tell the candidates apart from that record. A cause in a product rule stops the task (NEEDS_CONTEXT).
3. If it is the worker race: the two arms that leave the rail collapsed for the next app wait until the config the next app loads (`load_settings()`) records the collapse, instead of a fixed 0.3 s pause. Leave the witness, fold-hint and painted-frame comparisons unchanged.
4. Run the test 10+ times under `-n 4`, plus the whole file; record any fold-hint mismatch on TASK-37000.5.
5. Preflight; task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause: the worker-shutdown race, not the retained seed, and not a product rule.**

A scratch plugin (loaded through `PYTEST_PLUGINS`, since the private-profile child drops `-p` and `PYTEST_ADDOPTS`; deleted) recorded each rail save's dispatch, start and end. At each Console mount it also recorded the retained seeds, the in-memory and on-disk records and the earlier app's save-worker states.
- Both apps select the same key, `console_rail_state:global:shared-layout-v1`.
- Each config save took 0.3-0.8 s under `-n 4` (load 30-50), longer than the 0.3 s pause.
- With the collapse save slowed by 3 s inside its thread, the dev test failed 4 of 4 runs at `assert not rail.display`, the CI assertion. The second app's loaded record held `right_open = True`, the first app's close worker was `CANCELLED`, and its thread still finished the write afterwards. The first app's exit does not wait for the thread, and the second app takes `load_settings()`'s cache, which the save republishes only after writing.
- Seed-only control: with every rail save dropped, the dev test passed 2 of 2. The second app started from the retained first-use seed alone, which has no `right_open` key, so a seed cannot open the rail. Where a record exists (all real runs), the explicit record wins anyway.
- Natural reruns at dev under `-n 4`: the sibling `test_an_inspector_opened_after_chat_switches_matches_one_open_throughout` failed 1 of 8 at `assert not rail.display`, the same race in its own leave-collapsed step. The target test passed 48 of 48.

In production, a started save thread completes before the process exits, so the race needs a second app in the same process. That is the test, not a product rule.

**Fix** (`Tests/UI/test_console_collapsed_inspector_send_layout.py`): a new `_leave_collapsed` helper collapses the rail and waits until `load_settings()`, which is what the next app loads, records `right_open = False` under the rail's persistence key. Both arms that leave the rail for a next app use it, replacing the 0.3 s pause. The witness-row, fold-hint and painted-frame comparisons are unchanged.

**Red/green.** The slowed-save run failed 4 of 4 at dev and passed 4 of 4 with the fix. With every rail save dropped, the new wait fails ("the collapse never reached the config the next app loads"), so it cannot pass vacuously.

**Runs at this head** (`-n 4`, load 15-50):
- The target test: 12 of 12, 24 of 24, and 7 of 8 in 8 whole-file repeats. All 44 started collapsed. The one failure was the painted-frame check (`set(frames)`: the opened rail's first two frames showed the pre-send rows); see the follow-up below.
- The chat-switch test: 8 of 8.
- The whole file plainly: 3 passed.
- Preflight passed.

**AC#4.** No fold-hint mismatch recurred in 44 runs at this head or 48 at dev. The earlier local instance (1 of 12 at 88af9f9379) is now recorded on TASK-37000.5 with this test's evidence.

**Proposed follow-up (not filed):** the painted-frame assertion failed once in 44 runs at this head under load about 50, and not in 48 runs at dev. This change runs only after each arm's witness and frames are taken.

**Lesson:** `lessons-testing-evidence.md`, on waiting for the first app's save before booting a second app.

**Files:** `Tests/UI/test_console_collapsed_inspector_send_layout.py`, `backlog/tasks/task-37000.5 - ...md`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
