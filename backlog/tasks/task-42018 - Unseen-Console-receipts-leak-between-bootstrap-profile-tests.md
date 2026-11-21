---
id: TASK-42018
title: "Unseen Console receipts leak between bootstrap-profile tests"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:52
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r17cn-rev3 (round 17). Every `bootstrap_profile` test in one process shares one ChaChaNotes database (`Tests/conftest.py` ~62, ~1906-1908), and TASK-36478 restores only config.toml between them. A mounted send that closes before its reply renders leaves an unseen receipt, so the next app announces "A Console turn completed while hidden". TASK-41900/41904 fixed their own tests locally.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Durable Console unseen receipt marks are cleared between bootstrap-profile tests, as config.toml is
- [x] #2 A test shows a planted unseen receipt does not reach the next test
<!-- AC:END -->

## Implementation Plan

1. Add a teardown step beside `_restore_bootstrap_profile_config` in `isolate_test_environment`: for bootstrap-profile tests, delete only the Console unseen-receipt marks and their outcome companions (the rows `acknowledge_console_unseen` deletes) from the shared ChaChaNotes database, located without reading config.
2. Add writer/reader probes to `Tests/test_bootstrap_profile_config_reset.py` (TASK-36478's child-pytest pattern): one plants an unseen receipt, the next must not see it. Launch them from a test in a file a PR lane already runs.
3. Show the reader red without the clear, then green; run the config-reset tests and the receipt-sensitive Console tests.

## Implementation Notes

- **Clear (`Tests/conftest.py`).** New `_clear_bootstrap_console_receipts()` runs in `isolate_test_environment`'s teardown, in a `finally` after `_restore_bootstrap_profile_config`, so it covers the same tests (bootstrap profile, not a private-profile child) and still runs when the restore raises. It deletes only the rows `acknowledge_console_unseen` deletes, `console_unseen:%` marks and their `console_terminal_outcome:%` companions (prefixes taken from `ConversationLocalMarksService`); chats, stars and every other mark stay. No need to clear the whole database.
- **Where the database is.** Found from HOME (`profile_paths.selected_default_base_data_dir()` / `*/<chachanotes leaf>`), not through config: a config read at teardown could create a config file the next test's restore then treats as its own. A probe showed the shared file at `<bootstrap root>/home/.local/share/tldw_cli/default_user/tldw_chatbook_ChaChaNotes.db`, absent until a test creates it, so tests that never open it pay one glob. A plain `sqlite3` connection with a 30 s busy timeout, closed at once; a file with no schema yet (`no such table`) is skipped. `conversation_local_marks` has no triggers or foreign keys. I checked for bootstrap tests that `monkeypatch` `sqlite3.connect` or the `profile_paths` helpers, whose patch would outlive this teardown (TASK-42028's lesson): none.
- **Test (AC#2).** Probes `probe_g_leaves_an_unseen_console_receipt` / `probe_h_sees_no_unseen_console_receipt` in `Tests/test_bootstrap_profile_config_reset.py` use TASK-36478's child pytest process, so they run in one process in file order whatever xdist does. `g` plants an unseen receipt with its "complete" outcome (`set_console_terminal_with_cursor`, what a finished turn writes) and stars the same chat. `h` asserts no unseen receipt, no outcome, and the star still there. That file runs in no PR lane, so the launcher `test_an_unseen_console_receipt_never_reaches_the_next_bootstrap_test` sits in `Tests/UI/test_console_send_without_mcp_store.py`, a whole UI census entry where TASK-41900 hit this leak (no census edit).
- **Red/green (temporary edits, reverted).** Clear disabled: `probe_h` failed with `assert (('task-42018-chat', '42018000-...'),) == ()`; with the clear it passes. Clear widened to delete every mark: `probe_h` failed (only its star assertion can fail then), so the narrow clear is pinned too.
- **Covering runs.** One `-n 4` command over the config-reset file and the bootstrap/receipt Console files (`Chat/test_console_chat_controller.py`, `test_console_durable_turn_acceptance.py`, `Performance/test_console_attention_owned.py`, UI `test_console_archive_recovery_flow.py`, `test_console_poll_reconciliation.py`, `test_console_visible_turn_attention.py`, `test_console_send_without_mcp_store.py`, `test_console_runtime_ownership.py`, `test_console_send_draft_snapshot.py`, `test_console_send_acknowledgement.py`). With this change: 30 failed / 604 passed (load 15-21). Same command with this task's files at HEAD: 30 failed / 603 passed (load 18-36). 28 reds are shared (draft snapshot, durable-turn cancellation, archive RAG matrix, poll reconciliation, visible-turn attention). Three draft-snapshot nodes red in only one run or the other are a flaky family that is already red. Red only with the change: `test_console_archive_recovery_flow.py::test_restore_as_requires_fresh_final_name_confirmation`. It uses its own tmp-path stores and no notices, passed 4 of 4 alone (load 32-34) and in both whole-file runs at `-n 4` (load 27-34; only the file's 3 RAG-matrix reds failed), so it is a load flake, not this change.
- **Review rev1 (fixes).**
  - *Critical:* `Tests/UI/test_library_shell.py::test_details_recomputes_db_sizes_on_open` (bootstrap) writes 4096 zero bytes to the shared ChaChaNotes path. The clear then raised `sqlite3.DatabaseError: file is not a database` at that test's teardown and at every later bootstrap test's in the process. That node, the next one in its file and the two ownership nodes, serially with the HEAD conftest: 4 passed, 4 errors. The clear now reads the 16-byte header and skips anything that is not `SQLite format 3\0` (an empty file too), and opens with `file:...?mode=rw` (`uri=True`), so a file deleted meanwhile is never created; a missing file or one that cannot be opened is skipped. Same command now: 4 passed.
  - New probes `probe_i_leaves_a_file_that_is_not_a_database` / `probe_j_runs_after_a_file_that_is_not_a_database` plant those zero bytes and check that the next test passes and the bytes are untouched. The launcher now runs g-j and requires 4 passed and no error. Red first, before the fix: `4 passed, 6 deselected, 2 errors` (`file is not a database` at both teardowns); green after.
  - *Minor:* `probe_h` asserted `console_terminal_outcome(...) is None`, which holds whenever the unseen mark is gone. It now SELECTs `console_terminal_outcome:%` rows directly and requires none. Control: the clear narrowed to `console_unseen:%` alone fails `probe_h` with `['console_terminal_outcome:42018000-...:complete'] == []`.
  - Rerun of the covering command after these fixes, with `test_library_shell.py`'s zeroing node and the node after it checked serially as above: 31 failed / 603 passed (load 14 -> 38). The 10 reds outside `test_console_send_draft_snapshot.py` are exactly the base run's; the 21 draft-snapshot reds are all nodes that were red in one of the two earlier runs (TASK-41900 counted 21 there). `test_restore_as_requires_fresh_final_name_confirmation` passed.
- **Review rev2 (hardening).** Same failure class as the Critical: any teardown error here repeats on every later bootstrap-profile test. The clear now skips the file, silently, on any `OSError` from opening it (`PermissionError`, `IsADirectoryError`, a file deleted meanwhile) or any `sqlite3.DatabaseError` from connecting or deleting (no table yet, "attempt to write a readonly database", a valid header over a malformed body; `OperationalError` is a subclass). It still opens `mode=rw`, so it never creates a file. Probes g-j pass (g/h still prove a healthy database loses its receipts and keeps its star).
- **Lane time.** The launcher took 2.6 s (call) serially with the UI lane's flags.
- No lesson added: the mechanism follows TASK-36478's pattern. Follow-up proposed: TASK-41900/41904's local workarounds (`_acknowledge_leaked_receipts`, possibly `_wait_for_own_receipts_seen`) can go now.
- Files: `Tests/conftest.py`, `Tests/test_bootstrap_profile_config_reset.py`, `Tests/UI/test_console_send_without_mcp_store.py`.
