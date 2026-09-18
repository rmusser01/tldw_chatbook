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

| Root drafts, ordered saves and truthful scope | TASK-32791, [targeted checks and ten native captures](../qa/2026-09-18-mcp-root-settings/README.md) | Exact draft ownership, app lifetime/shutdown, cache warnings and later config/external-file supersession; existing Permissions resize failure is separately retained |

## Remaining review

- **Permissions table final viewport:** TASK-32791's related rerun found the
  light-theme selected State cell clipped from 7 to 6 columns after resize. A fresh
  baseline checkout at6c0e317ab7 reproduced it unchanged on the first attempt;
  an isolated current retry passed. The plain Permissions DataTable observes
  canvas resize but not all final child/outer-scrollbar geometry transitions.
  Repair and qualify this separately; passing retries do not close it.
- **Local-tools master toggle:** verify ordering, lifetime and truthful outcomes
  under overlapping saves; the root-specific owner does not change this path.
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
