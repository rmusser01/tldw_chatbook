# MCP component review

This ledger separates qualified repairs from remaining destination behavior.
Token migration alone does not qualify a complete workflow.

## Qualified

| Surface | Evidence | Bounds |
| --- | --- | --- |
| Settings → permission Edit | TASK-32785, [129 targeted cases and native gallery](../qa/2026-09-18-mcp-permission-handoff/README.md) | Exact profile/revision, visible controls/rows and read-only canvas scrolling |
| Compact introduction, Source and permission matrix | TASK-32788, [134 targeted cases and native gallery](../qa/2026-09-18-mcp-compact-readability/README.md) | Complete guidance and Local/Server labels; permission Tool/State together, tags accessible; no connected external server qualification |
| Tools controls and catalog access | TASK-32789, [88 distinct targeted cases and native gallery](../qa/2026-09-18-mcp-tools-access/README.md) | Full on/off label, focused filters/rows, retained cursor through resize, real private toggle persistence and exact row inspection; no tool execution |
| Tools name/state readability and identity | TASK-32790, [88 final targeted cases and native gallery](../qa/2026-09-18-mcp-tools-readability/README.md) | Complete names and State together, metadata reachable, identity retained through resize/filter/refresh, independent Enter and short/long scrollbar transitions; no tool execution |

## Remaining review

- **Workspace-root guidance and save lifecycle:** independent mounted review
  confirmed three defects: copy promises next-Console confinement, ordinary
  `_sync_children()` discards the root draft, and a cancelled `to_thread` save A
  can finish after save B and leave disk at A while the UI claims B. Actual
  configured-root consumers include standalone MCP serving and the operator Hub
  executable-provider/test path (`unified_control_plane_service.py`); Console
  uses Chat scratch and admitted Workspace folders. Blank fallback resolves the
  serving process's current working directory. Correct this contract and qualify
  ordered writes, draft retention, failure/retry and screen lifetime.
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
