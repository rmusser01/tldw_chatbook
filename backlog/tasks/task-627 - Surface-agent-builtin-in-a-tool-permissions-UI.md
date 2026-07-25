---
id: TASK-627
title: Surface agent:builtin in a tool permissions UI
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [tools, security, ux, mcp]
dependencies: [TASK-545]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-545/P1 (`Docs/superpowers/specs/2026-07-25-builtin-tool-permission-gate-design.md`) added a permission gate for agent-runtime built-in tools under the namespace `server_key = "agent:builtin"`, resolved via `resolve_builtin_state`. It deliberately restricts decisions to session scope (approve-once / approve-for-session) and never writes a persistent allow/deny, because the MCP workbench (`UI/MCP_Modules/mcp_workbench.py`) renders rows from the live MCP catalog — a synthetic `agent:builtin` key does not appear there, so a persistent decision recorded for it could not be undone in-app short of hand-editing the permission-store JSON.

P2 of TASK-545 will port mutating fs/note tools (`write_file`, `create_note`, `update_note`, ...) behind this same gate and tag them `risk_tags=("mutates",)`, at which point users will want to set a durable allow/deny instead of re-approving every session. Persistent decisions must not ship until there is a UI surface that can list and reverse them. This task adds that surface — either a dedicated row/section in the existing MCP workbench (Permissions mode) for the `agent:builtin` "server", or a lightweight Tools settings pane — so `always_allow`/`deny` become safe to offer for built-in tools.

Also file here: the current bulk Approve-all/Deny-all wiring on `ChatApprovalCard` (task-545/T6) is row-aware in the sense that it skips applying a bulk value a row's narrowed `options` cannot accept (e.g. a built-in row has no `always_allow`), but a row left unresolved by a bulk click currently gives no visual signal that it still needs a manual decision. Any UI work here should also make that a visible, not silent, state.

Also note: a cancelled built-in approval (stop/unmount mid-approval) is already recorded into the MCP execution log today — `ConsoleChatController._record_cancelled_approval_decisions` (`console_chat_controller.py:1108`) is owner-agnostic and calls `service.record_tool_decision(call.server_key, ...)` for any still-pending call, built-in or MCP, so a cancelled built-in call writes an `agent:builtin` row there. No permission state is written and the call never raises, so this is arguably desirable audit parity — but it means a synthetic "server" can already appear in an MCP-labelled log, which the UI work here should account for (e.g. when listing/filtering execution history by server).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] `agent:builtin` and its tools are listed somewhere in Settings/MCP UI with their currently effective state (allow/ask/deny) and origin
- [ ] A user can set and clear a persistent allow or deny for an individual built-in tool from this UI, and the change is visible immediately (no restart required)
- [ ] The UI clearly distinguishes `agent:builtin` (in-process agent-runtime tools) from `builtin:tldw_chatbook` (the built-in MCP server) so the two namespaces are never presented as one
- [ ] A bulk Approve-all/Deny-all action that cannot apply its value to a given row (because that row's `options` excludes it) leaves that row visibly flagged as still needing a decision, not silently unchanged
- [ ] This task is a documented prerequisite for TASK-545/P2 offering persistent (not just session-scoped) decisions for built-in tools
<!-- AC:END -->
