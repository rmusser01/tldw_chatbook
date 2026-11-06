---
id: TASK-38960
title: "Durable-commit quiet-gap tests count only syncs that reach the tick projection, so a sync deferred under load reads as held"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- tests
- flaky
- console
priority: medium
references:
- Tests/UI/test_console_durable_commit_quiet_gap.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. The UI PR lane runs `Tests/UI/test_console_durable_commit_quiet_gap.py` as a whole file.

**Runs through the scratch-profile wrapper:**
- whole file with `-n 4`: 10 of 10 passed, at load 34-65;
- the first test alone: 5 of 5 passed, at load 33-40;
- whole file serial (`-p no:xdist`): 3 of 11 failed, at load 31-50:
  - `test_a_durable_commit_runs_its_first_sync_then_holds_until_the_next_tick`, at load 35 (the assertion text was not captured);
  - `test_without_a_run_poll_a_durable_commit_holds_no_sync`, with `assert 0 == 1` on `_Syncs.count`, at load 47;
  - `test_a_durable_commit_runs_its_first_sync_then_holds_until_the_next_tick` again, with `AssertionError: the poll's tick ran no sync` (`assert 0 == 1`), at load 48.

**Likely cause.**
- `_Syncs` (test file `:54-67`) counts only the `config_sync.project_console_tick` calls made in the test's own task.
- A whole sync can return before reaching that call: `prepare_console_presentation_sync` refuses (`chat_screen.py:17853`, `:17863`; `config_sync.py:109`), for example while config warm-up is refused under load.
- The sync then schedules a delayed replay (`full_sync_replay_delay`, TASK-38696), which runs in another task the test does not count. So a sync the test expects reads as "ran no sync" or "held".

**History and dedup.** The file last changed in 03938ede0b (TASK-36610) and in 63cb61436a and 354da741bc (TASK-33620.15.4). TASK-36963 (the `_POLL_SYNC` mark is never reset) is a different defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All four tests in the file pass 20 of 20 serial whole-file runs and 10 of 10 -n 4 whole-file runs, at a recorded load of at least 40
- [ ] #2 A sync whose presentation preparation refuses is either kept out of the hold tests' window or counted on purpose, and the test says which; shown by forcing one refusal under a temporary source edit
- [ ] #3 Each hold assertion still fails when the hold is removed (_held_for_durable_commit always returns False), shown under a temporary source edit
<!-- AC:END -->
