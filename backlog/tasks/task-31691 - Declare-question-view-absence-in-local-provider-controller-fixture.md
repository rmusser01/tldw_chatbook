---
id: TASK-31691
title: Declare question-view absence in local-provider controller fixture
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:32'
updated_date: '2026-09-05 18:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep local provider composition tests honest about their headless question capability after the controller protocol grew.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Five reported local-provider composition tests pass with scratch and todo assertions preserved
- [x] #2 The bare controller fixture explicitly represents no question-view capability
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only initialization of an existing optional view callback.
1. Reproduce the missing set_pending_question failures and compare real initialization.
2. Add the missing no-view callback declaration to the bounded fixture.
3. Run the complete local-review-hook test file and static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initialized the bare ConsoleChatController fixture with set_pending_question=None, matching the real constructor optional view callback. Scratch ownership, approvals, and todo assertions are unchanged. Five missing-protocol RED cases reproduced and fixed. Full48-file GREEN in the installed review environment:48 passed10.24s. The shared editable venv produced three misleading scratch-unavailable results because isolated workspace-executor subprocesses loaded another checkout; an escalated rerun still failed until the installed review environment was used. No production authority fallback added. Ruff lint, changed-block formatting and diff checks passed; self-reviewed. No new ADR for this test-only initialization.
<!-- SECTION:NOTES:END -->
