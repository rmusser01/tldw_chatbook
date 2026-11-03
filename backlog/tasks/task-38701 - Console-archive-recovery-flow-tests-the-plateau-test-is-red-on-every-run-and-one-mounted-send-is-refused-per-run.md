---
id: TASK-38701
title: "Console archive-recovery flow tests: the plateau test is red on every run and one mounted send is refused per run"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_archive_recovery_flow.py
- tldw_chatbook/UI/Console_Modules/archive_recovery.py
dependencies:
- TASK-38652
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down. Checked on dev 9bbd5c2515, 2026-11-03: two whole-file runs with `-n 4` (load 24-25) each gave 2 failed and 41 passed.
- `test_retained_native_recovery_resource_owners_plateau_and_close` failed in both runs at `assert stable[-1]["live_handles"]["chat"] >= 2` (`assert 1 >= 2`).
- A different mounted-recovery test failed in each run with `_ConsoleTurnRefusedError: Console turn was refused before durable acceptance.` (console_runtime.py:3388): `test_native_result_refresh_remains_manual_and_imported_json_is_inert` in run 1 and `test_native_date_search_labels_monday_chat_friday_workspace_and_unknown[False]` in run 2.

The file is in no PR lane (not in the UI census, no workflow target), so neither failure has been reported. PR #3366 (TASK-38652) edits archive_recovery.py's sync, which these tests drive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_retained_native_recovery_resource_owners_plateau_and_close passes on dev, with the reason it saw one live chat handle where it expects two recorded
- [ ] #2 The refused-turn failures are explained, and the file passes 5 of 5 whole-file -n 4 runs on dev
- [ ] #3 The file runs on a PR lane, or the task notes record why it does not
<!-- AC:END -->
