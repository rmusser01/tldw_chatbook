---
id: TASK-38649
title: Console controls can stay stale when a cold readiness refresh returns early
status: To Do
created_date: 2026-11-03 05:51
labels:
- follow-up
- console
- bug
priority: medium
references:
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- Tests/UI/test_console_readiness_config_projection.py
- Tests/UI/test_console_send_ready_attachment.py
dependencies:
- TASK-38645
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-38645 (2026-11-03); not yet shown to be reachable.

When the readiness display is cold, `ConsoleReadinessConfigProjection.run` (console_spend_projection.py:384 on dev 18482c1cf7) starts a refresh worker and returns False without asking for a retry. The refresh (`_refresh`, :479) has several early returns: the screen is closing, the key moved during the read, the source or owner changed, or an exception (`except Exception: return`, :612). On those paths it neither publishes a whole sync nor asks for a retry. A lone `_sync_console_control_bar` that hit a cold display could then leave Send and the Workbench actions in their old state until some unrelated trigger runs another sync.

Whole syncs re-warm through `prepare_console_presentation_sync`, and `retry_after_config_rebuild` schedules a 0.2 s retry for some callers, so the gap may already be closed in practice. The task is to find out, and close it if it is open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test or probe establishes, for each early-return path of a cold refresh, whether the Console controls stay stale with no further trigger
- [ ] #2 If any path leaves them stale: that path requests a retry or a sync, so the controls converge on their own; a test pins it and is shown red before the fix
- [ ] #3 If no path does: the evidence is recorded in the notes and no product code changes
- [ ] #4 A closing screen or disposed runtime still never schedules more work
<!-- AC:END -->
