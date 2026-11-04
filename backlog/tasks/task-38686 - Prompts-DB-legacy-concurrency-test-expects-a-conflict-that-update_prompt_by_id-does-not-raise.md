---
id: TASK-38686
title: "Prompts DB legacy concurrency test expects a conflict that update_prompt_by_id does not raise"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: '2026-11-04 01:19'
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
- [ ] #1 Two concurrent updates of one prompt that pass the same expected_version produce exactly one success and one conflict, and the rewritten test pins that, passing 10 of 10 runs alone on dev
- [x] #2 The test fails when update_prompt_by_id's version comparison is removed (shown under a temporary source edit)
- [x] #3 The test runs on a PR lane, or the task notes record why it does not
<!-- AC:END -->

## Implementation Plan

1. Decide test-or-product from the code and history. `update_prompt_by_id` compares versions only when the caller passes `expected_version` (added in 2026-08 for the Library prompt editors). The test predates that (2025-06, 7ebdcc0731) and never passed one, so the test is what is wrong.
2. Rewrite the test: two writers on their own connections both hold the version they opened, and both save it as `expected_version`. Count only `ExpectedVersionConflictError` as the conflict, and pin one success, one conflict, version 2, and the winner's details.
3. AC#1 asks for concurrent saves. Probe how overlapping transactions resolve before choosing how the test orders them.
4. AC#2: remove the `expected_version` comparison under a temporary source edit and show the test red. AC#3: record the lane outcome. This brief allows workflow edits only for TASK-38699.

## Implementation Notes

**The test was wrong, not the product.** `update_prompt_by_id` has compared versions only when the caller passes `expected_version`, added 2026-08 for the Library prompt editors (`prompt_save_ack.py`, `prompt_history.py`, `library_prompts_controller.py`). Without one, an update is last-writer-wins by design. The test dates from 2025-06 (7ebdcc0731) and never passed `expected_version`. Its "conflict" could only come from SQLite lock errors, which it counted as conflicts (`"locked" in str(e)`).

**Rewrite (committed).** Two editors open version 1. Each saves on its own connection and thread, passing `expected_version=opened`, and the saves run one after the other. Only `ExpectedVersionConflictError` counts as the conflict. The test pins `{1: "success", 2: "conflict"}`, version 2, and the first writer's details.
- 10 of 10 runs alone, each its own process through run.sh at load 15-19, about 1 s each.
- AC#2: with the `expected_version != current_version` check deleted (temporary edit, reverted), the test is red with `{1: 'success', 2: 'success'} != {1: 'success', 2: 'conflict'}`.
- The whole file gives 70 passed (`-n 4`).

**AC#1 left open: overlapping saves do not end in a conflict.** A scratch probe ran two threads behind a barrier, each passing the same `expected_version`, on a fresh file DB per round. The loser got one of three errors:
- `ExpectedVersionConflictError`: 106 of 300 rounds at load 34, 31 of 100 at load 28.
- `ConflictError("Prompt update lost a version race")`, from SQLITE_BUSY_SNAPSHOT 517: 8 and 2 rounds.
- `DatabaseError("Failed to update prompt ID 1: database is locked")`: 186 and 67 rounds. Its cause is a plain SQLITE_BUSY (5) on the deferred transaction's read-to-write upgrade, while the winner holds the write lock. `_is_busy_snapshot_error` maps only 517 to `ConflictError`.

The winner was the only success in every round, so no update was lost. But with overlapping transactions, AC#1's "one success and one conflict" holds in only about a third of the rounds under load. A test that pins it with truly overlapping threads cannot pass 10 of 10 without a product change, either of:
- (a) map SQLITE_BUSY on this update's write to `ConflictError`, as 517 already is, so a racing save reads as "changed elsewhere" rather than "database is locked";
- (b) begin this update with `BEGIN IMMEDIATE`, so the later writer waits for the lock and then fails the version comparison.

Both change what a user sees on a racing save, so the choice is left to the owner. The committed test pins sequential saves of one opened version, which is the expected_version contract. AC#1 stays unticked until the owner decides whether that meets it.

**AC#3: not on a PR lane, by this brief's rule.** The only Prompts_DB lane target is `test_prompt_export_mixins.py` in `derived-artifacts.yml`, and this burn-down allowed workflow edits only for TASK-38699. The whole legacy file is now green (70 tests, about 6 s), so it could be gated whole. That is listed as a proposed follow-up.

Files: `Tests/Prompts_DB/test_prompts_db_legacy.py`, `backlog/docs/lessons-testing-evidence.md` (new entry on overlapping SQLite writers).

