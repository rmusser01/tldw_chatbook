---
id: TASK-40320
title: "All 11 archive nodes of test_workspace_archive_async_completion.py are red on dev, cause unknown, and the file runs on no lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 05:45
labels:
- follow-up
- tests
- workspaces
- dev-red
priority: low
references:
- Tests/UI/test_workspace_archive_async_completion.py
dependencies:
- TASK-39169
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream review r13p-rev1: 11 failed at base ae9f9e8d21 and at the cprod head). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Every archive node of `Tests/UI/test_workspace_archive_async_completion.py` fails, on both the Console and Settings surfaces; the unarchive node passes. The tests drive the real handlers through a `SimpleNamespace` `_host` (`:20-56`), so drift in that stand-in is the first suspect, but the cause has not been read. The file runs on no lane, so archive's async completion has no guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the 11 nodes is classified (test drift with the commit, or a product defect fixed with a test that was red) and passes at `-n 4`
- [x] #2 The file runs on a PR lane
- [x] #3 test_workspace_archive_review_regressions.py's three Console restore nodes pass (stand-in run_worker accepts an async callable) and the file runs on the UI lane
<!-- AC:END -->

## Implementation Plan

1. Re-run the file at -n 4 on base 26b2e677dd and read every node's error.
2. Trace each error to the product line it hits and the commit that added that line; decide whether the product or the stand-in is wrong (the real `WorkspaceRecord`, registry signature and `ChatScreen` are the reference).
3. Fix the drifted stand-ins in the test file only; fix the product only if it is wrong.
4. Append the file to the UI census tail, raise `MINIMUM_FILES`, and record its local serial time with the lane's flags.
5. Re-run the file at -n 4 until green; record classification per node.

## Implementation Notes

All 11 nodes were test drift: the `_host`/stand-in doubles fell behind three product commits. The product is right in each case, so only the test file changed.

**Classification (each against the real collaborator):**
- 9 nodes, drift from dd8d60c9e8 (#3162, merged 2026-10-31): archive now passes `expected_archive_revision=record.archive_revision` (the real `WorkspaceRecord` has `archive_revision: int = 0`). The stand-in records had no `archive_revision` and the `storage(_)`/`archive(_)` doubles took no keyword, so `_complete_archive` raised before the write. Nodes: `test_workspace_archive_dispatches_read_and_write_offloop[console|settings]`, `test_workspace_archive_failure_never_releases_another_reservation[False-console|False-settings]`, `test_workspace_archive_cancellation_keeps_reservation_until_commit[False|True-console|settings]` (4), `test_cancelled_archive_receipt_appears_after_late_dialog_dismissal`.
- 2 nodes, drift from b6859aee5f (#3173, merged 2026-10-31): the resume holds `self._screen._session.conversation_activation_window()` (ChatScreen's session controller). The stand-in `_screen` was `object()`. Nodes: `test_late_existing_resume_keeps_authority_through_token_preparation[True|False]`.
- `test_cancelled_archive_receipt_appears_after_late_dialog_dismissal` also drifted from d043e22438 (#3480, merged 2026-11-11): Console hands `run_worker` the bound `_sync_native_console_chat_ui` uncalled, which Textual accepts; the stand-in `run_worker` took only coroutines.

**Fix.** A `_record()` helper with `archive_revision=0`; the storage doubles take `**_`; `_screen` carries `_session.conversation_activation_window = contextlib.nullcontext`; the stand-in `run_worker` calls an async callable as Textual does.

**Evidence.** Base 26b2e677dd, `-n 4`: 11 failed, 13 passed (load about 33). After: 24 passed in 3 of 3 runs at `-n 4` (load about 72).

**Lane.** The whole file is appended to `scripts/ui_pr_gate_census.txt` (`MINIMUM_FILES` = measured count). Local serial time with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`): 24 passed in 23.3 s (35.7 s wall), load 65-70 on 18 CPUs.

**Review r14set-rev1 (AC #3).** `Tests/UI/test_workspace_archive_review_regressions.py` had the same d043e22438 drift in `_recovery_host`. Its stand-in `run_worker` now calls an async callable as Textual does. Red at base 26b2e677dd with `TypeError: a coroutine was expected`: `test_console_restore_invalidates_cached_rows_before_sync`, `test_console_restore_storage_does_not_block_ui_loop`, `test_console_receipt_undo_read_failure_has_async_recovery`. After: 23 passed at `-n 4`. The whole file is appended to the census tail (`MINIMUM_FILES` = measured count). Serial with the lane's flags: 23 passed in 16.5 s (22.4 s wall), load about 33. Its five `ASYNC110` ruff findings were there at base; I left them alone.

Files: `Tests/UI/test_workspace_archive_async_completion.py`, `Tests/UI/test_workspace_archive_review_regressions.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
