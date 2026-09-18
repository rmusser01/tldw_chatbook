# MCP component review

This ledger separates qualified repairs from remaining destination behavior.
Token migration alone does not qualify a complete workflow.

## Qualified

| Surface | Evidence | Bounds |
| --- | --- | --- |
| Settings → permission Edit | TASK-32785, [129 targeted cases and native gallery](../qa/2026-09-18-mcp-permission-handoff/README.md) | Exact profile/revision, visible controls/rows and read-only canvas scrolling |
| Compact introduction, Source and permission matrix | TASK-32788, [134 targeted cases and native gallery](../qa/2026-09-18-mcp-compact-readability/README.md) | Complete guidance and Local/Server labels; permission Tool/State together, tags accessible; no connected external server qualification |
| Tools controls and catalog access | TASK-32789, [88 distinct targeted cases and native gallery](../qa/2026-09-18-mcp-tools-access/README.md) | Full on/off label, focused filters/rows, retained cursor through resize, real private toggle persistence and exact row inspection; no tool execution |

## Remaining review

- **Tools column readability:** final native 80-column captures show long Tool
  values moving State partly or entirely out of the leftmost viewport. Horizontal
  scroll remains available; simultaneous identity/state reading needs review.
- **Workspace-root guidance and save lifecycle:** `MCPToolsMode` still describes
  this root as applying to the next Console run, and
  `_save_tools_mode_workspace_root()` confirms confinement to it. This contradicts
  [ADR-102](../../../backlog/decisions/102-console-run-admitted-local-path-authority.md),
  which reserves configured-root fallback for the standalone local MCP server;
  in-app Console path tools derive authority from admitted workspace bindings.
  Validate actual consumers, then correct the user-facing contract and qualify
  save failure/retry and retained edits.
- **Tools refresh and execution:** retained selection/filter drafts across
  background refresh, diagnostic empty-state actions, disconnected/stale tools,
  schema forms/raw arguments, test execution and inspector recovery.
- **Servers:** source transitions, add/edit/remove and connection lifecycles,
  built-in enable/expose controls, errors and recovery.
- **Audit and remaining permissions:** filtering, exact tool/event drilldown,
  restored roots, bulk actions, review/approval and connected-runtime journeys.

The [component completion ledger](2026-09-17-design-system-completion-audit.md)
retains other destinations. Draft PR2707 remains open and subject to its own
visual review and merge approval.
