---
id: TASK-137
title: Enable Console workspace-scoped new chat creation
status: Done
labels:
- console
- workspaces
- uat
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Console's workspace rail support creating a new chat conversation in the active workspace so users can start work inside the current operating context without leaving Console. This closes the current gap where workspace switching works but new workspace conversation creation is explicitly deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can create a new Console chat from the active workspace context rail when a workspace is active.
- [x] #2 The created Console session is associated with the active workspace and appears in that workspace's conversation list.
- [x] #3 Switching workspaces updates the visible conversation list and does not leak workspace-specific conversations into other workspace rails.
- [x] #4 The Default workspace remains usable for normal chat creation while preserving the local-only/file-tools-disabled policy.
- [x] #5 Unavailable server sync/handoff paths remain explicitly labeled as WIP or unavailable and are not implied to run.
- [x] #6 Mounted regression coverage verifies active-workspace creation, Default workspace creation, and cross-workspace visibility boundaries.
- [x] #7 Rendered CDP/Textual-web evidence is captured before approval/PR completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare `codex/console-workspace-new-chat` against `origin/dev` and identify the focused workspace-new-chat changes to port into the current workspace.
2. Verify the current Console display state exposes the workspace rail `New conversation` action for Default and named active workspaces while preserving server/handoff unavailable copy.
3. Consolidate mounted regression coverage so Default workspace creation, active workspace creation, and cross-workspace visibility boundaries are each collected by pytest.
4. Confirm the workspace rail action routes through native Console session creation and assigns the active workspace to new sessions without enabling sync/server handoff paths.
5. Run focused Console/workspace tests, capture fresh rendered CDP/Textual-web evidence, then update acceptance criteria and implementation notes.

ADR required: no
ADR path: N/A
Reason: ADR 005 already establishes the local-first Console workspace/server-readiness boundary. This task is bounded UI/session wiring and regression coverage; it does not introduce storage/schema changes, sync policy changes, service-contract changes, or new data ownership boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Inspected `codex/console-workspace-new-chat` against refreshed `origin/dev`; the focused branch commit enabled the workspace rail `New conversation` action, routed it through native Console session creation, and added Default/active-workspace mounted regressions.
- Confirmed the current workspace already contained the production behavior from that branch: the rail action calls `_create_native_console_session_from_active_context()`, new sessions inherit the active workspace settings/context, and `build_console_workspace_state()` exposes `new_conversation_enabled=True` for Default and active workspaces without enabling sync/server handoff.
- Consolidated `Tests/UI/test_console_native_chat_flow.py` by removing the duplicate `test_console_workspace_rail_new_conversation_creates_default_workspace_session` definition so the Default workspace mounted regression is collected exactly once, while keeping the native-flow assertion focused on session creation and removal of the deferred-creation blocker.
- Captured fresh rendered Textual-web/CDP evidence from the rebased `codex/task-120-workspace-new-chat` worktree and documented it in `Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-chat-evidence-2026-06-22.md`:
  - `Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-conversation-initial-cdp-2026-06-22.png`
  - `Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-conversation-after-click-cdp-2026-06-22.png`
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console workspace rail new-chat creation is enabled and verified in the current workspace. The Default workspace can create a new local chat from the rail while keeping sync/file tools disabled and server handoff copy explicitly unavailable; named active workspaces create sessions scoped to that workspace; switching workspaces keeps workspace-specific conversations isolated.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_list_reserves_two_line_rows_with_margin Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_appears_in_workspace_conversation_rail Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_promotes_active_native_session_in_workspace_rail Tests/UI/test_console_native_chat_flow.py::test_console_workspace_rail_new_conversation_creates_default_workspace_session Tests/UI/test_console_native_chat_flow.py::test_console_workspace_rail_new_conversation_stays_scoped_to_active_workspace Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_row_switches_native_session Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_switch_restores_transcript_messages --tb=short` -> `38 passed, 8 warnings in 28.77s`
- `git diff --check` -> no output
- `file Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-conversation-initial-cdp-2026-06-22.png Docs/superpowers/qa/console-uat-parallelization/task-120-workspace-new-conversation-after-click-cdp-2026-06-22.png` -> both PNG image data, 1280 x 720

ADR required: no
ADR path: N/A
Reason: ADR 005 already establishes the local-first Console workspace/server-readiness boundary. This task only wires an existing UI action/session path and mounted regression coverage.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
