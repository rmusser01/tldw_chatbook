---
id: TASK-38686
title: "Prompts DB legacy concurrency test expects a conflict that update_prompt_by_id does not raise"
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: '2026-11-04 04:08'
labels:
- follow-up
- tests
- database
priority: low
references:
- Tests/Prompts_DB/test_prompts_db_legacy.py
- tldw_chatbook/DB/Prompts_DB.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down, where it was reported failing 6 of 6 runs alone. Checked on dev 9bbd5c2515, 2026-11-03.

`Tests/Prompts_DB/test_prompts_db_legacy.py::TestConcurrencyAndDataIntegrity::test_concurrent_updates_to_same_prompt` failed 5 of 5 runs alone, each its own pytest process through the scratch-profile wrapper (about 1 s each), with `AssertionError: 'conflict' not found in dict_values(['success', 'success'])`.

**Why it fails.** The test starts two threads behind a barrier. Each calls `update_prompt_by_id(pid, {"details": ...})` without `expected_version`, and the test asserts that one of them raises a conflict and that the final version is 2. `update_prompt_by_id` (Prompts_DB.py:1818) compares versions only when the caller passes `expected_version` (:1890). Without it, each update reads the current version inside its own transaction (:1870-1887), so two serialized updates both succeed.

**No lane runs it.** The only Tests/Prompts_DB target on a PR lane is `test_prompt_export_mixins.py` (derived-artifacts.yml:376). The core-tests job in test.yml runs on pushes to main only, and Nightly Deep last ran on 2026-09-27.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Two concurrent updates of one prompt that pass the same expected_version produce exactly one success and one conflict, and the rewritten test pins that, passing 10 of 10 runs alone on dev
- [x] #2 The test fails when update_prompt_by_id's version comparison is removed (shown under a temporary source edit)
- [x] #3 The test runs on a PR lane, or the task notes record why it does not
- [x] #4 Overlapping `add_prompt(overwrite=True)` saves of one prompt both land, one after the other (the later overwrite wins, version goes up by 2), never a raw lock error or a lost-race conflict.
- [x] #5 The Console mix-in Save runs its prompt write off the event loop, so a contended save cannot freeze the UI
<!-- AC:END -->

## Implementation Plan

1. Decide test-or-product from the code and history. `update_prompt_by_id` compares versions only when the caller passes `expected_version` (added in 2026-08 for the Library prompt editors). The test predates that (2025-06, 7ebdcc0731) and never passed one, so the test is what is wrong.
2. Rewrite the test: two writers on their own connections both hold the version they opened, and both save it as `expected_version`. Count only `ExpectedVersionConflictError` as the conflict, and pin one success, one conflict, version 2, and the winner's details.
3. AC#1 asks for concurrent saves. Probe how overlapping transactions resolve before choosing how the test orders them.
4. AC#2: remove the `expected_version` comparison under a temporary source edit and show the test red. AC#3: record the lane outcome. This brief allows workflow edits only for TASK-38699.
5. Owner decision (coordinator, 2026-11-04): option (b). `update_prompt_by_id` begins its transaction IMMEDIATE, so an overlapping save waits for the write lock and then fails the version check. Add an overlapping-saves test for AC#1, and show it red with the BEGIN deferred again.
6. AC#4 (coordinator, 2026-11-04): `add_prompt` opens `transaction(immediate=serialize_create or overwrite)`. Add a deterministic overlapping-overwrite test, red with the deferred transaction, and rerun the probe for the overwrite path.
7. AC#5 (review, 2026-11-04): IMMEDIATE turns an instant SQLITE_BUSY into a wait of up to the 10 s busy timeout, and the Console mix-in Save ran its write on the event loop. Move `PromptScopeService.save_prompt`'s local create/update to a worker thread (`asyncio.to_thread` + `run_finite_local_worker`, as `read_mixin_records` does). Pin it with a thread-recording spy that is red before the move.

## Implementation Notes

Five commits:
1. The test rewrite.
2. `update_prompt_by_id` begins IMMEDIATE, after the owner chose option (b).
3. The same fix for `add_prompt(overwrite=True)` (AC#4).
4. Review fixes: the wait claims below are corrected, and the Console mix-in Save's write moves off the event loop (AC#5).
5. Review round 2: an in-flight save can no longer be cancelled, and `:memory:` saves run inline.

**The test was wrong.** `update_prompt_by_id` has compared versions only when the caller passes `expected_version`. That parameter was added in 2026-08 for the Library prompt editors (`prompt_save_ack.py`, `prompt_history.py`, `library_prompts_controller.py`). Without it, an update is last-writer-wins by design. The test dates from 2025-06 (7ebdcc0731) and never passed `expected_version`. Its "conflict" could only come from SQLite lock errors, which it counted as conflicts.

**Rewrite.** `test_concurrent_updates_to_same_prompt`: two editors open version 1 and save it one after the other, each on its own connection and thread. Only `ExpectedVersionConflictError` counts as the conflict. The test pins `{1: "success", 2: "conflict"}`, version 2, and the first writer's details. It passed 10 of 10 runs alone. With the `expected_version != current_version` check deleted (a temporary edit), it is red with `{1: 'success', 2: 'success'} != ...` (AC#2).

**Overlapping saves were refused by SQLite's lock, not the version check.** A scratch probe raced two threads behind a barrier, both passing the same `expected_version`, on a fresh file DB per round. With the deferred BEGIN, the loser got:

| Loser's error | 300 rounds, load 34 | 100 rounds, load 28 |
|---|---|---|
| `ExpectedVersionConflictError` | 106 | 31 |
| `ConflictError("... lost a version race")` (SQLITE_BUSY_SNAPSHOT 517) | 8 | 2 |
| `DatabaseError("... database is locked")` (plain SQLITE_BUSY 5) | 186 | 67 |

The SQLITE_BUSY case happens when the loser's read-to-write upgrade meets the winner's write lock.

**Fix (owner decision: option b).** `update_prompt_by_id` now opens `self.transaction(immediate=True)` (`BEGIN IMMEDIATE`). This is the same mode the file's delete, restore and history-restore writers already use.
- The busy timeout is 10 s (`PRAGMA busy_timeout` reads 10000; `timeout=10` at connect). So a racing save waits for the lock, reads the committed version, and raises the `ExpectedVersionConflictError` the Library prompt UI already handles.
- The probe rerun at load 20-26 gave `('conflict', 'success')` in 300 of 300 rounds, with zero SQLITE_BUSY and zero 517.

**Nested callers.** `transaction()` issues no BEGIN when the connection is already in a transaction. So the one in-file nested caller, `restore_prompt_history_entry` (inside its own IMMEDIATE transaction), is unchanged, and no caller can be deadlocked by a second BEGIN.
- Every other caller (`update_prompt`, `Prompts_Interop.apply_server_prompt_version`, `local_prompt_service.update_prompt`, `prompt_scope_service.update_prompt`) calls it with no outer transaction.
- **IMMEDIATE adds a bounded wait** where the deferred path failed at once. Under the deferred BEGIN, the SELECT-then-UPDATE upgrade skipped the busy handler and failed with SQLITE_BUSY immediately: the review's probe failed in 0.00 s. IMMEDIATE waits for the write lock instead, for at most the 10 s busy timeout; in the same probe it waited 2.08 s and then succeeded.
- The review found one caller on the event loop: the Console mix-in Save (`ConsolePromptsModal._save_mixin_editor` → `_ConsolePromptMixinsFlow.save` → `PromptScopeService.save_prompt` → `LocalPromptService.update_prompt` → `update_prompt_by_id`). AC#5 moves that write off the event loop, so a contended save cannot freeze the UI.
- The 517 to `ConflictError` mapping is still reachable inside a caller's own deferred transaction.

**`test_busy_snapshot_from_a_second_writer_is_reported_as_conflict`** (`test_prompts_db_pytest.py`) ran the second writer's whole save from a trace callback at the first writer's UPDATE. Under IMMEDIATE, the first writer holds the lock by then, so that same-thread injection can only wait. The test now runs the update inside a caller's deferred transaction, where the race and its "version race" `ConflictError` still exist.

**New AC#1 test.** `test_overlapping_saves_of_one_opened_version_end_in_one_conflict`. Worker 1 holds its write transaction open after its UPDATE (a wrapped `_log_sync_event` waits for worker 2 to start, plus 0.3 s), while worker 2 saves the same opened version.
- 10 of 10 runs alone at load 13-28.
- Red 3 of 3 with the BEGIN deferred again (temporary edit): `{2: DatabaseError('Failed to update prompt ID 1: database is locked'), 1: 'success'} != {1: 'success', 2: 'conflict'}`.
- The ordering only decides whether the deferred control fails: under IMMEDIATE, worker 2 waits for worker 1's commit however the threads are scheduled.

**Runs** (`-n 4`, run.sh):
- `Tests/Prompts_DB`, `Tests/Prompt_Management`, `Tests/Library/test_library_prompts_seam.py`, `Tests/Console/test_prompt_draft_save.py` and `Tests/Chatbooks/test_chatbook_prompt_round_trip.py`: 53 failed, 1163 passed, 3 errors. With dev's `Prompts_DB.py` swapped in: 54 failed, 1162 passed, 3 errors. The node sets are identical apart from the new test, which is red on dev's source by design.
- The shared reds are on dev: `test_prompt_improvement_service.py` (50), `test_prompt_batch_models.py::test_prompt_management_lazy_public_exports_preserve_class_identities`, and two `test_prompts_db_batch_mutations.py` privacy-log nodes.
- Architecture ratchets (serial): 256 passed. `Prompts_DB.py`'s module-size row is re-pinned 4670 -> 4671 for the one-line comment (owner size waiver 2026-10-31).

**AC#4: the same race in `add_prompt(overwrite=True)`.** The overwrite path reads the version inside its own transaction and then writes `UPDATE ... AND version = ?`. It has no caller-supplied expected version.
- With the deferred BEGIN, 200 overlapping rounds at load 19 gave:
  - one `DatabaseError` "database is locked" (SQLITE_BUSY) in 186 rounds;
  - two successes, version +2, in 12 rounds;
  - one `ConflictError` "Prompt creation lost a WAL snapshot race" (517) in 2 rounds.
- The fix: `add_prompt` opens `self.transaction(immediate=serialize_create or overwrite)`. The probe then gave two successes in order, version +2, in 300 of 300 rounds at load 19-21, with no lock errors and no conflicts. The update probe rerun in the same session gave `('conflict', 'success')` in 300 of 300 rounds.
- "Both land in order" is the right outcome here: an overwrite replaces whatever is stored, as two overwrites that do not overlap always did. The AC wording was corrected on that basis.
- `overwrite=False` creates keep `serialize_create`'s mode. So deferred creates still map a 517 race to `ConflictError`, and `test_busy_snapshot_during_create_is_reported_as_conflict` is unchanged.
- Every `overwrite=True` path writes. IMMEDIATE turns a contended overwrite's immediate SQLITE_BUSY into a bounded wait for the write lock, at most the 10 s busy timeout. Nested callers are unaffected, because `transaction()` issues no BEGIN inside an open transaction.
- Test: `test_overlapping_overwrites_of_one_prompt_both_land_in_order`, with the hold-open harness now shared as `_overlap_two_saves`. The harness's catch is narrowed to `(DatabaseError, InputError)`, which removes a blind `except Exception` from the file.
  - 10 of 10 runs alone at load 19-20.
  - Red with the deferred transaction (4 of 4): `{2: DatabaseError("Failed to process prompt 'Overwrite Overlap': database is locked"), 1: 'success'} != {1: 'success', 2: 'success'}`.
  - After the refactor, the update overlap test passed 10 of 10 and was red 2 of 2 with its BEGIN deferred (`DatabaseError('... database is locked') is not an instance of ExpectedVersionConflictError`).
- Covering set (`-n 4`, same five targets as above):
  - Branch: 53 failed, 1164 passed, 3 errors.
  - Dev's `Prompts_DB.py` swapped in: 55 failed, 1162 passed, 3 errors.
  - The only difference is the two new overlap tests, both red on dev's source by design. The branch's failing tests are identical to the previous commit's run.
- Architecture ratchets (serial): 256 passed. `Prompts_DB.py` line count is unchanged (4671).

**AC#5: the Console mix-in Save no longer writes on the event loop.**
- **The change.** `PromptScopeService.save_prompt` now runs a local create or update in a worker thread: `asyncio.to_thread(run_finite_local_worker, ...)`, the pattern `read_mixin_records` and `count_prompts` already use. A coroutine-returning local service is completed with `asyncio.run` inside that worker, as `count_prompts` does. Server saves are unchanged.
- **Thread safety.**
  - `PromptsDatabase` keeps one connection per thread (`threading.local`, `check_same_thread=False`), so the worker opens its own, and `run_finite_local_worker` retires it when the worker ends.
  - **Correction (review round 2):** that is wrong for `:memory:`. The worker's own connection is a fresh, empty database, and the review's probe failed with `no such table: Prompts`. An earlier version of these notes said the `:memory:` suites passing showed this was safe; none of them writes through `save_prompt`.
  - A `:memory:` save now runs inline (`prompt_db.is_memory_db`). `test_prompt_scope_service.py::test_local_save_over_an_in_memory_database_creates_and_updates` (create, then a versioned update) was red with that error before the change and is green after.
- **Test.** `Tests/UI/test_console_prompt_mixins.py::test_mixin_save_writes_the_prompt_off_the_event_loop` drives the mounted mix-in editor's Save twice and records `threading.get_ident()` in a spy on `update_prompt_by_id`.
  - The first save lands. The editor closes and focus goes to `console-prompts-close`, the focus measured while the save still ran on the loop.
  - For the second save the spy raises `ExpectedVersionConflictError` inside the worker. The status shows the same "Library changes were not saved ..." copy, and the edited name is kept.
  - Before the change it was red only on the thread assertion: `assert 8461753728 not in [8461753728, 8461753728]`. After it: 5 of 5.
- **Mix-in and Library prompt UI files** (`-n 4`, same command, 24 UI files plus `Tests/Chat/test_console_prompt_mixin_*.py` and `Tests/Library/test_library_prompts_seam.py`):
  - Before the change: 291 failed, 723 passed. After: 289 failed, 726 passed (including the new test).
  - Both runs are dominated by the local RecoveryRequired setup trap (TASK-38111), 271 nodes.
  - The failing sets differ only by load flakes. Three nodes failed only before the change, and one only after: `test_collection_mutation_close_requests_dispatch_without_queuing[close_sources1-create]`, which passed 27 of 27 in three runs alone afterwards.
- **`Tests/Prompts_DB` + `Tests/Prompt_Management`** (`-n 4`): 53 failed, 1123 passed, 3 errors both with and without the change, the same nodes, all red on dev.
- `Tests/CI`: 1198 passed, 1 skipped. Architecture ratchets (serial): 256 passed.

**Review round 2 (2026-11-04): a press during a save could cancel it.**
- **The bug.** Moving the write into a thread gave the save worker a point where it suspends mid-write. Save and Preview both start exclusive `console-mixins` workers, so either one pressed during a contended save cancelled the save after its write landed. The preference reconcile and the editor's saved state never ran. In the review's probe, 4 of 4 rounds left the DB at version 2 while the editor kept version 1 as unsaved, and the next Save was refused.
- **The fix** is the smaller option. While a mix-in save worker is unfinished, Save and Preview presses are ignored. Either press, and the save itself, shows "Saving Library changes…" in the workbench status line, which the save's outcome then replaces. Focus is not moved.
- **Regression test:** `test_press_during_contended_mixin_save_cannot_cancel_it[save|preview]`. A raw connection holds `BEGIN IMMEDIATE`, and a spy on `update_prompt_by_id` shows the first save has reached its write before the second press. The test pins:
  - the "Saving" copy;
  - after release: DB version 2, the editor closed, and the preferences reconciled to the saved name;
  - a further Save landing as version 3.
- **Red at 0d12515a18**, with the "Saving" assertion temporarily removed so the cancel itself shows:
  - `[preview]` failed 3 of 3 with "the editor kept version 1";
  - `[save]` failed 2 of 3. It passes when the second save's thread wins the lock, so its own conditional write lands.
- **Green:** 5 of 5 runs. The review's probe now gives version 2 and a closed editor in 8 of 8 rounds.
- **Other save paths, checked.**
  - The Console prompt editor's saves (`_save_prompt`, `_save_recipe`, `_update_original`, and the Draft Shelf buttons in `_shell_button`) are awaited inside `@on` handlers, which the modal's message pump runs one at a time. They are not exposed; confirmed.
  - The system-prompt modal's Save to Library was exposed. It starts an exclusive worker per press, so a second press during the threaded write cancelled the first and saved again. It gets the same guard: the press is ignored while the first save runs, and the first save's outcome copy lands.
  - `test_console_system_prompt.py::test_second_save_to_library_press_cannot_cancel_an_in_flight_save` was red before the change (`['Formal Tone', 'Formal Tone'] == ['Formal Tone']`) and passed 3 of 3 after.
- **Lane.** The AC#5 off-loop pin, the regression node and the Save to Library node are appended at the UI census tail, with the floor at 1376. Measured serially at load 20: 10 s, 13 s (both cases) and 6 s.
- **The review's 14 UI files plus `test_console_system_prompt.py`** (`-n 4`, same command): 241 failed, 510 passed on the branch; 243 failed, 508 passed with the three changed sources at 0d12515a18.
  - Failing only with 0d12515a18's sources: the three new tests, `test_prompt_system_replacement_refreshes_visible_status_chip[append_active]`, and one `test_literal_history_restore_keeps_kind_definition_and_guards` case.
  - Failing only on the branch:
    - `test_browse_mixins_read_does_not_hold_navigation[close]`, which also failed in the earlier base run;
    - another `test_literal_history_restore_...` case. That test fails 2-3 of 9 cases alone on both trees;
    - `test_literal_save_keeps_live_defining_module_policy_callbacks`, which failed at its "Save did not reach its source worker" wait, before any real save runs, and passed 3 of 3 alone afterwards.
  - Most failures on both sides are the local RecoveryRequired setup trap (TASK-38111).
- `Tests/Prompts_DB` + `Tests/Prompt_Management`: 53 failed, 1124 passed, 3 errors, with the same failing tests as before (all red on dev). `Tests/CI`: 1198 passed, 1 skipped. Ratchets: green.

**AC#3: not on a PR lane, by this brief's rule.** The only Prompts_DB lane target is `test_prompt_export_mixins.py` in `derived-artifacts.yml`, and this burn-down allowed workflow edits only for TASK-38699. The legacy file is green (72 tests, about 5 s), so it could be gated whole (proposed follow-up).

Files: `tldw_chatbook/DB/Prompts_DB.py`, `tldw_chatbook/Prompt_Management/prompt_scope_service.py`, `tldw_chatbook/Widgets/Console/console_prompts_modal.py`, `tldw_chatbook/Widgets/Console/console_system_prompt_modal.py`, `Tests/Prompts_DB/test_prompts_db_legacy.py`, `Tests/Prompts_DB/test_prompts_db_pytest.py`, `Tests/Prompt_Management/test_prompt_scope_service.py`, `Tests/UI/test_console_prompt_mixins.py`, `Tests/UI/test_console_system_prompt.py`, `Tests/Architecture/test_module_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md`.
