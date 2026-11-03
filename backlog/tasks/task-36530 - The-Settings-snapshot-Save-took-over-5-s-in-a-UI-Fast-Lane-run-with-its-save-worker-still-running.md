---
id: TASK-36530
title: The Settings snapshot Save took over 5 s in a UI Fast Lane run, with its save worker still running
status: To Do
updated_date: 2026-11-02 21:52
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

**More occurrences (recorded 2026-11-02).** Searching 62 failed UI-job logs from 46 derived-artifacts runs (2026-10-21 to 11-02) found four more failures. Each was `[size1]` with a child `TimeoutError`, in UI Fast Lane (25):

| Run | Attempt | Branch | Date | Outcome |
|---|---|---|---|---|
| 40909862501 | 1 | codex/task-35582-delivery | 2026-10-29 | green on rerun |
| 41003666368 | 1 | fu/h-railstate (#3225) | 2026-10-30 | green on rerun |
| 41189968143 | 1 | fu/h-railstate (#3225) | 2026-10-31 | green on rerun |
| 41224394415 | 1 | fu/h-railstate (#3225) | 2026-10-31 | stayed red |

The same set holds 19 UI (25) logs: the 4 above failed this node and 15 passed it. Of the 16 UI (25) jobs on other branches, 1 failed it. `[size0]` is not a census entry. Four of the five known failures, counting the first, came from #3225's branch, which changed Console rail-state persistence (TASK-36385). On 2026-10-31 that branch failed the node in 2 of 4 attempts:
- 41189968143: attempt 1 failed, attempt 2 passed;
- 41224394415: attempt 1 failed;
- 41248007404 (the merge-time run, on d3dc1c99b0): attempt 1 passed, with a 20.14 s call.

A local paired probe found no difference that #3225 caused:
- The branch added no config writes to this test.
- The snapshot save took 147-235 ms on the branch and 131-275 ms on dev.
- The test passed 4 of 4 runs on both trees with all 18 CPUs loaded.
- The branch's UI (25) shard ran 600 s, inside the 528-814 s of other runs.

The owner finished #3225 with three more commits (retained rail seeds: b787009712, d02046bf61 and edd6260383, which are 3098e0705c, 71ddbecdbb and d3dc1c99b0 on dev) and merged it on 2026-10-31. The final code ran twice in CI: 41224394415 on edd6260383 failed, and the merge-time run passed. No run on dev since then has been checked. If it fails again, instrument it in CI: log the save thread's executor wait and lock wait from a temporary debug commit. Local runs have not reproduced it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's failure note says where the save's time went (executor queue, lock wait, write, resume)
- [ ] #2 The cause of the CI stall is identified and recorded; if Save can be held behind other work in the app, that is fixed or recorded as accepted with a reason
- [ ] #3 The test passes 20 of 20 serial runs, and its time bound rests on measured save times
<!-- AC:END -->
