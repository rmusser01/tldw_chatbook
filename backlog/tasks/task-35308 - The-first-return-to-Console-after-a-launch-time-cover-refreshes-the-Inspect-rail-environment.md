---
id: TASK-35308
title: 'The first return to Console after a launch-time cover refreshes the Inspect rail environment'
status: To Do
assignee: []
created_date: '2026-10-17 09:11'
labels:
- model-config-redesign
- follow-up
- console
dependencies:
- TASK-33008
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/environment.py
- Tests/UI/test_console_local_server_discovery_card.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by PR #3144's review round 5 (TASK-33008, 2026-10-16). The app pushes the first-run wizard with `call_after_refresh` right after the first Console push, so the wizard suspends Console before its attach reconciliation finishes. On the first return, view-start re-claims the one-shot mount token and the resume reads as the mount's own (`mount_already_refreshed` is true). Round 6 made discovery re-arm on that return (`_console_discovery_rearm_pending`, set in `on_screen_suspend`).

At origin/dev 111f8e48fc the same `on_screen_resume` still calls `self._console_environment_owner.notify_rail_opened()` only when `not mount_already_refreshed`, so on that first return the Inspect rail's environment owner is never told the rail is open, and neither tier refreshes. `sync_task_resume_state()` sits behind the same condition. A server or workspace change made while the wizard was open is therefore not shown on that return.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On the first return to Console after a screen covered it at launch (the first-run wizard), the Inspect rail's environment owner gets `notify_rail_opened`; pinned by a test in the style of `test_a_card_covered_at_launch_runs_discovery_again_on_the_first_return`
- [ ] #2 An ordinary mount's own resume still refreshes once (task-15475's dedupe holds)
- [ ] #3 ChatScreen gains no method and chat_screen.py stays within its size row
<!-- AC:END -->
