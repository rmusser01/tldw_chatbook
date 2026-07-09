---
id: TASK-125
title: Enable Console workspace create and switch UAT path
status: Done
assignee: []
created_date: '2026-06-18 15:43'
updated_date: '2026-06-18 16:31'
labels:
  - console
  - workspaces
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console and Library Workspaces surfaces expose a usable local-first path for creating or selecting an explicit workspace, while keeping the safe Default workspace available for ordinary chat and keeping server sync/handoff paths honestly unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can create or expose an explicit local workspace from the visible workspace-management UI.
- [x] #2 Console Change workspace is reachable when more than one workspace exists.
- [x] #3 User can switch active Console workspace and visible conversation rows update without cross-workspace leakage.
- [x] #4 User can create a new Console chat scoped to the active workspace.
- [x] #5 Default workspace remains usable for normal chat creation with file tools disabled.
- [x] #6 Server sync and handoff paths remain labeled WIP or unavailable.
- [x] #7 Mounted regressions cover clean-profile creation or availability, workspace switching, active-workspace chat creation, and cross-workspace visibility boundaries.
- [x] #8 Rendered CDP/Textual-web evidence is captured before PR completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/005-console-workspace-server-readiness.md
Reason: ADR-005 already defines the local-first Console workspace boundary, Default workspace safety policy, and sync/handoff non-goals for this slice.

1. Add failing mounted regressions for the visible workspace creation/switching path and active-workspace new conversation behavior.
2. Enable the Console workspace rail's New conversation action for local workspaces, including Default, using the existing native Console session store and workspace registry memberships.
3. Add or expose the minimal Library Workspaces action needed to create/select an explicit local workspace without implying server sync.
4. Keep server readiness, ACP handoff, and sync copy explicitly unavailable/WIP in visible state.
5. Run focused Console/Library workspace tests, then capture textual-web/CDP screenshots for the clean-profile path.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Enabled the existing Console workspace rail `New conversation` affordance for local workspaces, including the safe Default workspace, and wired it to create a native Console chat session scoped to the active workspace. Added a visible Library > Workspaces `Create local workspace` action that creates `Workspace 1`, makes it active, refreshes the left-rail workspace scope label, and keeps server sync/handoff copy explicitly WIP/unavailable.

Added mounted regressions for Default workspace conversation creation, active workspace chat creation, Library local workspace creation, and stale Library scope-label prevention. Verified with:

`python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_appears_in_workspace_conversation_rail Tests/UI/test_console_native_chat_flow.py::test_console_workspace_rail_new_conversation_creates_default_workspace_session Tests/UI/test_console_native_chat_flow.py::test_console_send_after_workspace_switch_persists_to_selected_workspace Tests/UI/test_console_native_chat_flow.py::test_console_workspace_switch_refreshes_visible_session_tabs Tests/UI/test_post_release_workspaces_library_depth.py --tb=short`

Result: 19 passed, 1 dependency warning from `requests`.

Review follow-up: the fallback state where the registry exists but no active workspace is selected now keeps Default conversation creation enabled, and Console provider/session selection restores the safe Default workspace before creating or sending a chat. Verified the follow-up with:

`python -m pytest -q Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py::test_console_provider_selection_reads_local_llamacpp_configured_model Tests/UI/test_console_native_chat_flow.py::test_console_provider_selection_restores_default_workspace_when_none_active Tests/UI/test_console_native_chat_flow.py::test_console_new_chat_tab_appears_in_workspace_conversation_rail Tests/UI/test_console_native_chat_flow.py::test_console_workspace_rail_new_conversation_creates_default_workspace_session Tests/UI/test_console_native_chat_flow.py::test_console_send_after_workspace_switch_persists_to_selected_workspace Tests/UI/test_console_native_chat_flow.py::test_console_workspace_switch_refreshes_visible_session_tabs Tests/UI/test_post_release_workspaces_library_depth.py --tb=short`

Result: 39 passed, with existing dependency/deprecation warnings.

Qodo follow-up: local workspace identity generation now considers archived workspace records so `workspace-local-N` IDs and `Workspace N` names are not reused after archival, and the new Library create handler has a Google-style docstring. Added a mounted regression for archived `workspace-local-1` and verified the focused suite again.

Result: 40 passed, with existing dependency/deprecation warnings.

Rendered CDP/Textual-web evidence captured:

- `Docs/superpowers/qa/product-maturity/screen-qa/console/console-workspace-default-new-conversation-current.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/library-workspaces-create-local-before-current.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/library-workspaces-create-local-after-current.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/console-workspace1-before-new-conversation-current.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/console-workspace1-after-new-conversation-current.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/console-change-workspace-activation-current.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/console-after-switch-default-current.png`

Residual UX note: after switching back to Default, previously opened Workspace 1 transcript tabs remain visible as inactive open tabs while the left workspace conversation rail correctly scopes to Default only. This preserves open-session recovery but should be reviewed in a future Console tab-management pass if stricter workspace visual isolation is required.
<!-- SECTION:NOTES:END -->
