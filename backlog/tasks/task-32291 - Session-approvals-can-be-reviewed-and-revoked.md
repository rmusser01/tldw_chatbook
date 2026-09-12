---
id: TASK-32291
title: Session approvals can be reviewed and revoked
status: Done
assignee: []
created_date: '2026-09-10 19:18'
updated_date: '2026-09-11 06:31'
labels:
  - mcp
  - approvals
  - permissions
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Approve for session' is held in memory with no caller of the clear path; the only way to revoke it is restarting the app, and nothing shows which tools are session-approved. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Session-approved tools are listed (inspector or Permissions) with a revoke action.
- [x] #2 Revoking makes the next call to that tool ask again.
- [x] #3 Docs state that a session approval lasts until Chatbook exits or is revoked.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace both session-approval stores and the task-32281 arg-rule inspector/matrix pattern.
2. Failing tests: list/revoke round-trip on the service and the built-in gate; matrix ` (session)` suffix; inspector group + Revoke.
3. Implement list/revoke on both seams, the matrix suffix + legend, the inspector group and the workbench revoke handler.
4. Run the covering files against the pre-change baseline; preflight; docs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a per-entry review/revoke path for in-memory 'Approve for session' grants. Trace finding that shaped the design: there is only ONE store -- BuiltinToolGate.stamp() writes built-in grants into the SAME UnifiedMCPControlPlaneService._session_approvals set under agent:builtin (the gate is a facade wired to app.unified_mcp_service, not a second store), so the workbench reads the service alone and 'merging both' would have double-rendered every built-in row. Service gained list_session_approvals(profile_id) (sorted (server_key, tool_name) pairs) and revoke_session_approval(server_key, tool_name, profile_id) -> bool; the gate gained the same pair as a built-ins-only, profile-scoped view (a non-builtin server_key is refused there, never forwarded). No permission-store fence on revoke -- it only ever removes a permission, so a stale profile digest cannot make it unsafe. UI mirrors task-32281's arg-rule flow: the Permissions matrix State cell gains a ' (session)' text suffix (both the MCP and built-in row builders, appended after the rule markers so _perm_row_kind()'s leading-word read is untouched) with the legend gaining '(session) approved until Chatbook exits'; the inspector's permission container renders a 'Session approvals' group -- every live grant in the profile, not just this row's tool, since nothing listed them anywhere before -- each with a Revoke button posting the ROW's own pair via a new RevokeSessionApprovalRequested message. The workbench handler revokes, resyncs the matrix, and re-renders the open block through a new MCPInspector.refresh_permission_session_approvals(), which replays the block's own cached inputs (tool/effective/cascade/goto/arg rules) rather than re-resolving against the revoked tool -- that also keeps the built-in rows working, whose EffectiveToolState the inspector cannot reproduce. Tests (TDD, 10 new, red first): service list/revoke/profile-scoping round-trips; gate round-trip against a REAL UnifiedMCPControlPlaneService proving check() refuses again after revoke; inspector group rendering, absence, and per-row Revoke targeting; a workbench end-to-end covering the suffix on both an MCP and a built-in row, the listing, and revoke clearing suffix + row while the block stays open. Test-fixture note: the session-approval seam moved from ToolTestHubService up to the shared FakeHubService so PermissionsHubService inherits it. Files: MCP/unified_control_plane_service.py, Agents/builtin_tool_gate.py, UI/MCP_Modules/{mcp_inspector,mcp_permissions_mode,mcp_workbench}.py, Docs/User_Guide/mcp.md (new 'Session approvals' section), Docs/security/production-diagnostic-inventory.json (4 new warnings), Tests/{MCP/test_control_plane_permissions,Agents/test_builtin_tool_gate,UI/test_mcp_inspector,UI/test_mcp_permissions_mode,UI/test_mcp_workbench}.py. Covering suite 801 passed (baseline 791, +10 new, zero failures); preflight green. Whole-branch review round (R24, commit aa95995aad) fixed one folded minor here: list_session_approvals iterated self._session_approvals live, but an agent worker thread calls approve_for_session concurrently and a set mutated mid-iteration raises RuntimeError -- which MCPWorkbench._session_approvals_for_row swallows into an empty listing, blanking every ' (session)' suffix for that render. It now iterates a tuple() snapshot. Same commit added the mcp.md docs-pass stamp for this task (Session approvals section). Files: tldw_chatbook/MCP/unified_control_plane_service.py, Docs/User_Guide/mcp.md.
<!-- SECTION:NOTES:END -->
