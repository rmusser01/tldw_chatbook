---
id: TASK-32289
title: 'First-run tools copy: Quick setup and the first high-risk card'
status: Done
assignee: []
created_date: '2026-09-10 19:17'
updated_date: '2026-09-11 03:01'
labels:
  - onboarding
  - approvals
  - ux-copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Quick setup (six steps) never mentions tools; Full setup's tools step describes reads as safe while the first read_file card shows '(high risk)' with a hover-only explanation. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quick setup's summary tells the user that tools are off and where to enable them.
- [x] #2 The wizard description for read-class tools mentions the per-call approval.
- [x] #3 The high-risk reason on the card is visible without hover.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep existing tests for pins on the exact copy strings being changed (`_TOOL_COPY` descriptions, the Summary "Tools" row detail text) -- none found, safe to change.
2. Write failing tests: `_TOOL_COPY["read_file"][1]` (and the other 4 read-class tools) mentions per-call approval; `build_summary_rows(...)`'s "Tools" row detail reads the new destination string when no gates are on.
3. Implement: append " Asks you each time before running." to the five read-class tools' descriptions in `ToolsStep._TOOL_COPY`; change the "Tools" row's off-detail in `build_summary_rows` (`first_run_setup_state.py`) from "all off (default)" to "all off; turn them on under MCP ▸ Servers ▸ Tool gates".
4. Verify AC#3 is already satisfied by Task 5's `.approval-row-reason` Static (grep, no code change).
5. Run the two touched test files in full, compare against a pre-change baseline.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1/#2 required two copy-only edits; AC#3 was already satisfied by Task 5 on this branch (verified via grep for .approval-row-reason in chat_approval_card.py, no code change needed). ToolsStep._TOOL_COPY in FirstRunSetupWizard.py: appended " Asks you each time before running." to the five read-class tools' descriptions (read_file, list_directory, glob_files, grep_files, expand_document); the three mutating tools already carry a warning marker and were left alone. build_summary_rows in first_run_setup_state.py: changed the Tools row's off-state detail from "all off (default)" to "all off; turn them on under MCP Servers Tool gates" -- one string covers both Quick setup (which never shows the Tools step) and Full setup with every switch left off, since both share the same empty tools_on condition, so no track plumbing was needed. Added test_read_class_tool_copy_mentions_per_call_approval (Tests/Wizards/test_first_run_setup_wizard.py) and test_tools_row_off_detail_names_where_to_enable (Tests/Wizards/test_first_run_setup_state.py); confirmed RED against the unmodified implementation via a temporary git checkout revert, then GREEN. Ran both full test files (625 tests): 623 passed / 2 pre-existing failures, same names and same failure reason as the pre-change baseline (a stale expected-button-set pin unrelated to this change), so no regression. scripts/preflight.sh passed.

Qodo review follow-up (PR #2594, finding 6): that first sentence was unconditional and a tool-level Allow or a session approval makes it false, so the five read-class descriptions now end "Asks before running unless you approve a longer scope." and the pin above asserts the new sentence.
<!-- SECTION:NOTES:END -->
