---
id: TASK-32345
title: >-
  Show a waiting-for-approval state instead of Thinking or Generating while a
  card is pending
status: Done
assignee: []
created_date: '2026-09-10 19:11'
updated_date: '2026-09-11 01:24'
labels:
  - console
  - approvals
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With an approval card armed, the assistant row reads 'Thinking... <elapsed>', the status strip says 'Run: Agent running.', and the inspector says 'Live work: Generating...' and 'Run: Recovery required'. The activity table in UI/Console_Modules/agent.py has no waiting state; 'Waiting for approval' copy already exists unused in console_send_authority_summary.py. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While a round waits on the user, the assistant activity line, the status strip run chip and the inspector summary all say the run is waiting for the user's approval, with elapsed time.
- [x] #2 'Recovery required' and 'Generating...' never render for a healthy pending approval.
- [x] #3 Tests cover the pending-approval rendering of all three surfaces.
<!-- AC:END -->

## Renumbering provenance

<!-- SECTION:PROVENANCE:BEGIN -->
This task was filed as TASK-32276 on 2026-09-10 at 12:19 PT (19:19 UTC) in the
approval-card / MCP-permissions UX review wave (PR #2574), which makes it the
OLDER arrival against the TASK-32276 that `dev` minted later the same day
("Connect Console archive search and exact conversation resumption").

It renumbered to TASK-32345 anyway. The bare 2026-08-21 owner rule of TASK-19601
quoted in `scripts/check_backlog_task_ids.py` (older arrival keeps the id) is
superseded by the 2026-09-08 refinement recorded in
`backlog/docs/lessons-backlog-hygiene.md`: **landed-keeps-id trumps
older-keeps-id -- a task already on origin/dev never renumbers; the unmerged
side moves regardless of timestamps, because renumbering landed ids breaks
external references.** The dev-side TASK-32276 is merged and cited from
`backlog/decisions/147-conversation-archive-and-exact-resume.md`,
`backlog/docs/lessons-testing-evidence.md` and two plan/QA records; this task
was cited only from its own unmerged plan. `dev` applied the same refinement
earlier on 2026-09-10 when it renumbered its archive-lifecycle task to
TASK-32300.

Renumbered 2026-09-10. Commit messages on the wave branches
`approval-wave-a/b/c` written before this date that cite `task-32276` refer to
THIS task; the dev-side TASK-32276 keeps the id.
<!-- SECTION:PROVENANCE:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace all three surfaces to their real source instead of trusting the brief's line numbers: the activity line (`console_turn_activity_text` in `UI/Console_Modules/agent.py`), the run chip (`ChatScreen._console_active_run_copy` in `UI/Screens/chat_screen.py` — NOT `console_chat_controller.py`'s dispatch-start line, which is a one-time snapshot that never updates mid-turn), and the inspector (`console_send_authority_summary.py`'s existing `run = "Waiting for approval"` branch plus `console_display_state.py`'s "Live work" row).
2. Find why "Recovery required" fired instead: trace `_console_provider_blocker_copy` (feeds the Setup/Blocked-impact/Next-action inspector rows) and find it returns non-empty for ANY active run, including a healthy pending approval, because it does not exclude `recovery_action == "wait_for_active_run"` the way two sibling methods in the same file already do.
3. Write failing tests for all three surfaces (TDD) against the real code, using the controller's real `has_pending_approval_round`/`add_pending_round` registry, not a stand-in boolean.
4. Implement: a leading `pending_approval` branch in `console_turn_activity_text` (one constant, one branch, self-contained for a clean merge with lane A's sibling `setup` state); a `has_pending_approval_round` check in `_console_active_run_copy`; the `wait_for_active_run` exclusion in `_console_provider_blocker_copy`; and an `approval_count > 0` branch in the "Live work" row.
5. Run the named test files plus the full targeted regression set; verify only the pre-existing baseline reds remain.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Traced all three surfaces before touching code. The brief's suggested run-chip edit site (console_chat_controller.py's dispatch-start _set_run_state call, "Agent running.") is a one-time snapshot never re-evaluated mid-turn, so that surface's real fix landed in ChatScreen._console_active_run_copy instead, read fresh every 0.2s poll tick. Root cause for "Run: Recovery required": ChatScreen._console_provider_blocker_copy (feeds the Setup/Blocked-impact/Next-action Inspector rows) returned non-empty for ANY active run -- not just misconfigured providers -- because it never excluded recovery_action=="wait_for_active_run" the way two sibling methods in the same file already do; that produced a spurious "Next action" row on every active turn, which made console_send_authority_summary.py's existing (and correctly pinned-ordered) recovery_required check fire ahead of pending_approval_count. Fixed at that one root-cause function rather than reordering the projection's priority. Changes: UI/Console_Modules/agent.py adds a CONSOLE_TURN_ACTIVITY_WAITING_APPROVAL leading branch to console_turn_activity_text (one constant, one branch, self-contained for a clean merge with lane A's sibling setup state) plus getattr-guarded threading of has_pending_approval_round in ConsoleAgentController.console_turn_activity; UI/Screens/chat_screen.py adds the has_pending_approval_round check to _console_active_run_copy and the wait_for_active_run exclusion to _console_provider_blocker_copy; Chat/console_display_state.py's "Live work" row now checks approval_count before run_active. console_send_authority_summary.py needed no code change -- its existing branch was already correct and already pinned by a test; it just never fired. Docs/User_Guide/console/agent-runs-and-tools.md's Layout tour paragraph documents the new state. Tests added in test_console_turn_activity_line.py, test_console_status_chips.py, test_console_right_rail.py, test_console_display_state.py, all TDD RED before the fix, GREEN after. Verification: targeted six-file regression run, 148 passed / 4 failed (all 4 pre-existing, unrelated MagicMock-unpack and gateway-timing issues); full test_console_chat_controller.py run, 327 passed / 10 failed (all 10 pre-existing canvas durable-commit failures -- that file imports none of the three files this task touched, confirmed by grep).
<!-- SECTION:NOTES:END -->
