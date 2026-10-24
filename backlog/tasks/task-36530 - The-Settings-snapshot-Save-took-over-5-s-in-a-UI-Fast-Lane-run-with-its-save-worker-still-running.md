---
id: TASK-36530
title: The Settings snapshot Save took over 5 s in a UI Fast Lane run, with its save worker still running
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
- settings
- ci-flake
priority: medium
references:
- Tests/UI/test_llamacpp_snapshot_settings.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/LLM_Management/snapshot_settings.py
- tldw_chatbook/config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the CI run on PR #3225 (run 40024354185, UI Fast Lane (4), job 120656651965, 2026-10-23). `Tests/UI/test_llamacpp_snapshot_settings.py::test_f9_snapshot_controls_paint_and_keyboard_save[size1]` (140x45) is a UI census entry. It failed with `TimeoutError` at the 5 s `asyncio.timeout` around the save (:339-341), and so failed the required check. The lane runs the census serially, so the timeout did not come from xdist contention.

The failure note, added in c5897f76c4 after an earlier failure, shows the save was not lost but still running:
- before Enter: (focused, disabled, dirty, saving) = (True, False, True, False);
- after the timeout: (True, False, True, True);
- `loaded enabled: False`, `save workers: ['RUNNING']`.

So Enter reached the Save branch (settings_screen.py:33101-33116), but the worker's `asyncio.to_thread(save_snapshot_preferences, ...)` (:9392) had not returned after 5 s. `save_snapshot_preferences` is one `apply_settings_mutation_to_cli_config` call with a locked precondition (snapshot_settings.py:38-66).

On dev 5b5920b8e2 locally, a temporary timing probe (since reverted) measured that call 8 times, 4 of them in concurrent loaded runs at load average about 13. Each took 0.14-0.19 s, with no executor queue wait. A 5 s stall is about 30 times that, so this does not look like a wait that is too short for normal load. The cause is not identified. Candidates: the default executor queue, the config owner's lock, or the write itself on the CI runner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's failure note says where the save's time went (executor queue, lock wait, write, resume)
- [ ] #2 The cause of the CI stall is identified and recorded; if Save can be held behind other work in the app, that is fixed or recorded as accepted with a reason
- [ ] #3 The test passes 20 of 20 serial runs, and its time bound rests on measured save times
<!-- AC:END -->
