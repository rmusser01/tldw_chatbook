---
id: TASK-35303
title: 'The Inspector''s project-instruction status row tolerates a sync before its button mounts'
status: To Do
assignee: []
created_date: '2026-10-17 09:11'
labels:
- follow-up
- console
- flaky
dependencies: []
references:
- tldw_chatbook/Widgets/Console/console_project_instructions.py
- Tests/UI/test_console_local_server_discovery_card.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen in PR #3144's review round 5 (TASK-33008 notes, 2026-10-16): in one of twelve xdist runs of Tests/UI/test_console_local_server_discovery_card.py at load average about 40, `test_a_detected_legacy_alias_server_lists_its_models` failed before any suspend because a Console worker raised `NoMatches` for `#console-project-instruction-status-button` during mount. The test passes three of three serial runs; Phase 8 does not touch this code.

At origin/dev 111f8e48fc, `ConsoleProjectInstructionStatusRow.sync_state` (Widgets/Console/console_project_instructions.py) calls `self.query_one("#console-project-instruction-status-button", Button)` with no guard. A state sync that reaches the row after the row is mounted but before its `compose` has mounted the Button raises. The race was not reproduced on demand; the unguarded query still holds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A project-instruction state sync that arrives before the status row's button is mounted raises nothing, and the button shows the latest state once it mounts; pinned by a test that syncs before the row composes
- [ ] #2 `test_a_detected_legacy_alias_server_lists_its_models` no longer fails with this NoMatches under parallel load (repeated xdist runs recorded in the notes)
<!-- AC:END -->
