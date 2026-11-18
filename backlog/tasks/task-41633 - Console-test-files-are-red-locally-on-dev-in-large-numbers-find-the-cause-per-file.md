---
id: TASK-41633
title: "Console test files are red locally on dev in large numbers; find the cause per file"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/UI/test_console_internals_decomposition.py
- Tests/UI/test_console_send_draft_snapshot.py
- Tests/UI/test_console_workbench_contract.py
- Tests/UI/test_console_native_transcript.py
- Tests/Chat/test_change_turn_tracking.py
- Tests/UI/test_console_right_rail.py
- Tests/UI/test_console_rail_reconciliation.py
- Tests/Chat/test_console_async_mcp_snapshot.py
- Tests/MCP/test_console_snapshot_source_contracts.py
dependencies:
- TASK-41308
- TASK-41313
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** The burn-down's local runs (macOS, isolated profile through the scratch wrapper) found whole Console test files red on dev. These files are not on the PR lanes that ran, so nothing flags them. TASK-41313 fixed five such files: they failed behind the `raw_source_selection_changed` config-sandbox trap, and adding the `bootstrap_profile` marker (TASK-38111) cleared them. Some of the files below already carry that marker, so the trap is not the whole story.

**Red counts.** Figures from runs on dev 45b1949ca3 or 5e3c8b2bd4:
- `Tests/UI/test_console_internals_decomposition.py`: 122 (has the marker)
- `Tests/UI/test_console_native_transcript.py`: 110 (has the marker)
- `test_console_session_settings.py`: 97 (TASK-41308's run)
- `Tests/Chat/test_change_turn_tracking.py`: 46 (no marker)
- `Tests/UI/test_console_right_rail.py`: 33
- `Tests/Chat/test_console_async_mcp_snapshot.py`: 32
- `Tests/UI/test_console_rail_reconciliation.py`: 27
- `Tests/UI/test_console_send_draft_snapshot.py`: 20 of 33 (has the marker)
- `Tests/MCP/test_console_snapshot_source_contracts.py`: 16
- `Tests/UI/test_console_workbench_contract.py`: 12
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed file has a recorded cause in this task's notes: the marker trap, test drift, or a product defect
- [ ] #2 Files red only behind the config-sandbox trap pass with the `bootstrap_profile` marker
- [ ] #3 Every other cause is fixed here or filed as its own task with the failing node ids
<!-- AC:END -->
