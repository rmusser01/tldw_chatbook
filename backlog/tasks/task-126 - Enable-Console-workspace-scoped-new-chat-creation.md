---
id: TASK-126
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
1. Add mounted regressions proving the workspace rail exposes a new conversation action for Default and named active workspaces.
2. Add native Console flow coverage proving the action creates a new active session scoped to the current workspace.
3. Verify switching workspaces updates the workspace conversation rail and does not show other workspace-specific session rows.
4. Wire the existing native session creation path to the workspace rail action without enabling sync/server handoff paths.
5. Run focused Console workspace tests, capture rendered CDP/Textual-web evidence, then update task notes and DoD.

ADR required: no
ADR path: N/A
Reason: bounded UI/action wiring for an existing Console workspace/session model; no storage/schema, sync policy, service contract, or data ownership change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a workspace rail `New conversation` action that reuses Console's native session creation path, including Default workspace fallback and current settings propagation.
- Kept the Default workspace local-only/file-tools-disabled recovery copy visible; this change does not enable sync, server handoff, or filesystem tool authority.
- Added mounted regressions for Default and named workspace rail behavior, plus native Console flow coverage for workspace-scoped session creation and cross-workspace visibility boundaries.
- Captured rendered CDP/Textual-web evidence before completion:
  - `Docs/superpowers/qa/product-maturity/screen-qa/console/console-workspace-new-conversation-initial-cdp-2026-06-18.png`
  - `Docs/superpowers/qa/product-maturity/screen-qa/console/console-workspace-new-conversation-after-click-cdp-2026-06-18.png`
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console now lets users create a new native chat session directly from the Convos & Workspaces rail. The created session is scoped to the active workspace, appears in that workspace's conversation list, and remains isolated from other workspace rails.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_appears_in_workspace_conversation_rail Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_promotes_active_native_session_in_workspace_rail Tests/UI/test_console_native_chat_flow.py::test_console_workspace_rail_new_conversation_creates_default_workspace_session Tests/UI/test_console_native_chat_flow.py::test_console_workspace_rail_new_conversation_stays_scoped_to_active_workspace Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_row_switches_native_session Tests/UI/test_console_native_chat_flow.py::test_console_send_after_workspace_switch_persists_to_selected_workspace --tb=short`
- `git diff --check`

ADR required: no
ADR path: N/A
Reason: bounded UI/action wiring for an existing Console workspace/session model; no storage/schema, sync policy, service contract, or data ownership change.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
