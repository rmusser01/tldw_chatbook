---
id: TASK-656
title: Surface agent:builtin in a tool permissions UI
status: Done
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
- [x] `agent:builtin` and its tools are listed somewhere in Settings/MCP UI with their currently effective state (allow/ask/deny) and origin
- [x] A user can set and clear a persistent allow or deny for an individual built-in tool from this UI, and the change is visible immediately (no restart required)
- [x] The UI clearly distinguishes `agent:builtin` (in-process agent-runtime tools) from `builtin:tldw_chatbook` (the built-in MCP server) so the two namespaces are never presented as one
- [x] A bulk Approve-all/Deny-all action that cannot apply its value to a given row (because that row's `options` excludes it) leaves that row visibly flagged as still needing a decision, not silently unchanged
- [x] This task is a documented prerequisite for TASK-545/P2 offering persistent (not just session-scoped) decisions for built-in tools
<!-- AC:END -->

## Implementation Notes

Built per `Docs/superpowers/specs/2026-07-25-builtin-tool-permissions-ui-design.md`, informed by a comparative spike (Pi, CheetahClaws) recorded in that spec.

- **Hash-free namespace relaxation**: `HASH_FREE_SERVER_KEYS = frozenset({BUILTIN_TOOL_SERVER_KEY})` added to `MCP/permission_store.py`. Both `MCPPermissionStore.set_tool_state` and `UnifiedMCPControlPlaneService.set_tool_state` waive the `definition_hash`/`tool=` requirement for `state="allow"` only for keys in that set; every other `server_key` keeps the guard byte-identical, pinned by a test asserting MCP's behavior is unchanged and a test pinning the set's exact contents.
- **Enumeration without a run**: `builtin_permission_rows(payload) -> list[BuiltinPermRow]` in `Agents/builtin_tool_gate.py` resolves each catalog tool via `resolve_builtin_state` only (never `resolve_effective_state`/`effective_tool_states`, which apply MCP's ask-floor + hash check and set the `mark_config_changed()` rug-pull marker that `resolve_builtin_state` ignores). It also lists orphaned stored entries — a decision recorded for a tool a later release removed — so they stay visible and clearable rather than becoming permanently-set dead state.
- **UI section**: `MCPWorkbench` builds an `agent:builtin` section via a sibling method (`_builtin_permission_matrix_rows`/`_builtin_permission_rows`) and appends it to `rows` *after* `_build_permission_rows()` returns — `_resolve_effective_states()`/`_build_permission_rows()` have zero changed lines, so MCP resolution stays byte-identical. The section renders even with zero MCP servers configured, is labelled `Built-in (agent runtime)` distinctly from the MCP server's `builtin:tldw_chatbook`, and marks orphaned rows in the Tags cell while leaving `tool_name` undecorated for write-back. A fail-soft path keeps the pinned "Server default" row visible even if enumeration raises.
- **Persistent allow/deny fix**: `cycle_ui_state` is a strict ring Inherit→Allow→Ask→Off, so the first Space press on any built-in row landed on `allow`, which the pre-existing `_tool_for() is None` guard rejected with "Tool is no longer in the catalog" — wrong for a tool that is present, and it made `ask`/`deny` unreachable too since you could never step past the rejected `allow`. Fixed by branching on `BUILTIN_TOOL_SERVER_KEY` to skip the `HubTool` lookup and call `set_tool_state()` with no `tool=`, safe because of the `HASH_FREE_SERVER_KEYS` relaxation above. The MCP branch is untouched.
- **Bulk-action visual flag**: `_set_all_batch_decisions` now adds a `needs-decision` class to a row whose narrowed `options` excludes both bulk-button candidates; a new `@on(Select.Changed)` handler clears it once the row gets an explicit decision. Not reachable with any row shape shipped today (every current narrowing accepts a candidate from both bulk buttons) — this is a guard for future narrowings, exercised in tests via a synthetic row that excludes every candidate.
- **Fail-closed**: the constraint holds by construction, not by a dispatch guard. `_builtin_permission_matrix_rows` hard-codes `BUILTIN_TOOL_SERVER_KEY` and never accepts or branches on an externally supplied `server_key`, so there is no code path here that could resolve an unrecognized key by inheriting either branch's logic — there is nothing to dispatch on, because no catalog builder can mint an `agent:builtin` key in the first place (every MCP key is `local:`/`server:`/`builtin:`-prefixed). The spec's "fails closed on an unrecognized namespace" AC is therefore satisfied vacuously and has no meaningful test under this design; an unknown key reaching the MCP path still resolves via that path's own global default (`ask`), unrelated to this method.

This unblocks TASK-545/P2 from persistent-decision-scope-only: P2 can now tag mutating built-in tools `risk_tags=("mutates",)` and offer a durable allow/deny, because a persistent decision recorded here is visible and reversible in-app (the gap that made P1 restrict itself to session-scoped approvals only).

Modified/added: `tldw_chatbook/MCP/permission_store.py`, `tldw_chatbook/Agents/builtin_tool_gate.py`, `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`, `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`, CSS (`css/components/_agentic_terminal.tcss` + bundle), `Tests/Agents/test_builtin_tool_gate.py`, `Tests/UI/test_mcp_workbench.py`, `Tests/UI/test_console_mcp_approval.py`.

Known gaps not closed by this task (filed as follow-ups): the Permissions-mode inspector clears to empty when a built-in row is selected (`_tool_for()` only resolves `HubTool`s), and the rail's preview-sentence override count is computed on the pre-merge MCP-only `rows` before built-in rows are appended, so a built-in override changes the table but not the summary sentence.
