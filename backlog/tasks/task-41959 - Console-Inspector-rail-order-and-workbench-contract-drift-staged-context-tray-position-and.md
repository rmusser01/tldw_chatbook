---
id: TASK-41959
title: "Console Inspector rail order and workbench contract drift: staged-context tray position and five workbench nodes"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
- ui
priority: low
references:
- Tests/UI/test_console_workbench_contract.py
- Tests/UI/test_console_session_settings.py
dependencies:
- TASK-41633
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** The sub-agents section (`#console-agent-section-subagents`) now sits where the staged-context tray was pinned (`test_console_inspector_hosts_staged_context_above_source_readiness`, `test_console_left_rail_keeps_session_and_moves_staged_context_out`), and three more workbench-contract nodes are red on dev (TASK-41633 notes F3/F10).

**Failing nodes (dev d0486978dd, TASK-41633 runs).**
- `Tests/UI/test_console_workbench_contract.py::test_composer_menu_opens_current_draft_directly_in_recommended_review[size1]`
- `Tests/UI/test_console_workbench_contract.py::test_console_approvals_chip_activation_focuses_the_decision_not_submit`
- `Tests/UI/test_console_workbench_contract.py::test_console_blocked_inspector_explains_impact_and_next_action`
- `Tests/UI/test_console_workbench_contract.py::test_console_inspector_prioritizes_actionable_status_before_secondary_groups`
- `Tests/UI/test_console_workbench_contract.py::test_console_left_rail_keeps_session_and_moves_staged_context_out`
- `Tests/UI/test_console_session_settings.py::test_console_inspector_hosts_staged_context_above_source_readiness`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The intended Inspector order is confirmed (product) and the tests pin it
- [ ] #2 The workbench-contract nodes pass
<!-- AC:END -->
