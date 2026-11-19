---
id: TASK-38960
title: "Durable-commit quiet-gap tests count only syncs that reach the tick projection, so a sync deferred under load reads as held"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 05:17
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
- [x] #1 All four tests in the file pass 20 of 20 serial whole-file runs and 10 of 10 -n 4 whole-file runs, at a recorded load of at least 40
- [x] #2 A sync whose presentation preparation refuses is either kept out of the hold tests' window or counted on purpose, and the test says which; shown by forcing one refusal under a temporary source edit
- [x] #3 Each hold assertion still fails when the hold is removed (_held_for_durable_commit always returns False), shown under a temporary source edit
<!-- AC:END -->

## Implementation Plan

1. Reproduce serially under load with a temporary probe that logs each sync's hold decision, replay requests and preparation refusals, to see which path reads as "held".
2. Count on purpose at the hold's own decision (`begin_console_sync` not held, in the test's task) instead of at `project_console_tick`, and wait for any pending whole-sync replay before opening the commit window, so a replay in another task cannot take the commit's first sync.
3. Forced-refusal control (temporary source edit refusing one preparation inside the window): the new tests pass, the old counter fails.
4. Hold-removed control (`_held_for_durable_commit` always False): every hold assertion fails.
5. 20 serial and 10 -n 4 whole-file runs at load >= 40.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause, from a reproduction.** A temporary probe logged every `begin_console_sync` verdict, each replay request and each refused preparation. Run serially through the wrapper, the file failed 1 of 10 runs at load 44-90. That failure was `test_a_durable_commit_runs_its_first_sync_then_holds_until_the_next_tick` with "the commit held a second time" (`3 == 4`). The probe log showed:
- The hold had admitted the sync the test read as held.
- The tick before it ran `sync_console_core_state` while config was cold or a config lock was busy, and that called `replay_whole_console_sync` and set `_console_control_bar_replay_whole_sync`.
- The next explicit sync returned at that flag, before `project_console_tick`, so `_Syncs` never counted it.

A refused `prepare_console_presentation_sync` (the filed hypothesis) takes the same route. A deferring `_sync_console_rail_and_controls` sets the same flag.

**Fix (test only).** `_Syncs` now counts on purpose at the hold's own verdict: a `begin_console_sync` call in the test's task that returns not held. A held poll tick never reaches it, because `run_periodic_console_sync` skips that sync first. So the count is exactly what the hold let in, whether the sync then reaches the projection, has its preparation refused, or returns at a pending replay. The class docstring says this. Both window tests also wait, before opening the commit window, until no sync runs and no whole-sync replay is pending (`_quiet`). A replay runs in another task, and inside the window it could take the commit's first sync. No product code changed.

**AC#2 (forced refusal).** A temporary edit to `prepare_console_presentation_sync` refused the first preparation made while the store's durable commit window was open, once per screen:
- The fixed file passed 4 of 4, with two refusals forced.
- The base file failed both filed ways: "the commit's first tick was held" (`1 == 2`), and `assert 0 == 1` in `test_without_a_run_poll_a_durable_commit_holds_no_sync`.

**AC#3 (hold removed).** With `_held_for_durable_commit` returning False, a temporary copy of the file had every `assert` turned into a recorded check (AST rewrite), so that every assertion was evaluated. All 8 hold assertions in the first test failed: the explicit sync ran, result not False, no request left, the second tick synced, and the 3, 4, 6 and 7 counts were off. The 4 `held(...) is True` checks in `test_each_commit_holds_once_until_the_next_tick_and_never_two_ticks` also failed. The real file failed in both of those tests. The no-poll test has no hold assertion, and it stays green.

**AC#1.** Whole file, through the wrapper:
- Serial: 20 of 20 runs passed (4 passed each, 25-90 s). Nineteen runs started at a 1-min load of 40.8-69.4. The first started at 33.6 (it ended at 63.6), so I ran one extra serial run, which started at 42.9 and passed.
- `-n 4`: 10 of 10 runs passed (15-19 s), 9 of them starting at load 42.2-52.0 and one at 39.9. An extra `-n 4` run started at 42.4 and also passed.

Files: `Tests/UI/test_console_durable_commit_quiet_gap.py`. Lesson added to `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
