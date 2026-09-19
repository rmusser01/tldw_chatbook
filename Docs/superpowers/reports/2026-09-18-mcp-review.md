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
| Root drafts, ordered saves and truthful scope | TASK-32791, [targeted checks and ten native captures](../qa/2026-09-18-mcp-root-settings/README.md) | Exact draft ownership, app lifetime/shutdown, cache warnings and later config/external-file supersession; original Permissions resize failure retained and repaired below |
| Permissions final viewport and selected-row continuity | TASK-32792, [targeted checks and twelve native captures](../qa/2026-09-18-mcp-permission-reflow/README.md) | Child width/height reflow, full Tool/State paint, selected row/focus/filter retention and fresh-context Enter; no policy changes or tool execution |
| Shared local-tools master controls | TASK-32793, [177 distinct targeted cases and sixteen native captures](../qa/2026-09-18-mcp-master-settings/README.md) | Ordered saves across both entry points, activation/config identity, pending refresh/recreation, truthful partial receipts and shutdown; no runtime authority changes |
| Tools filter and focus continuity | TASK-32794, [127 distinct targeted cases and native gallery](../qa/2026-09-18-mcp-tools-refresh/README.md) | Open menu/highlight, real pointer/keyboard admission, delayed filter/drill ordering, empty/recovery focus; controlled empty projection, no connected-server or execution qualification |
| Workbench status lifetime | TASK-32795, [79 distinct targeted cases and eight native captures](../qa/2026-09-18-mcp-workbench-lifetime/README.md) | Pre-compose/prune polling, receipt replay across canvas/Workbench replacement, actual loading boundary; native navigation and saved receipts, no generic Select shutdown claim |
| Quiet table redraw and inspector clearing | TASK-32796, [275 distinct targeted cases and eight native captures](../qa/2026-09-18-mcp-table-selection/README.md) | Publication-time row/cell suppression, table-scoped gesture dedup and quiet external drills; real keyboard refresh/navigation, no schema/execution qualification |
| Compact rail navigation | TASK-32812, [119 targeted cases and eight native captures](../qa/2026-09-18-mcp-rail-navigation/README.md) | Full All servers/Source paint, literal Unicode names/counts, ordinary-refresh/resize focus, exact row identity and scrollbars; structural catalog replacement focus is not qualified |
| Complete tool-switch labels | TASK-32822, [37 targeted cases and eight native captures](../qa/2026-09-18-mcp-gate-labels/README.md) | All eleven gate labels, focus/scroll reveal, disabled dependencies and real private Deep research save/reversal; no broader save concurrency or execution qualification |

## Post-merge continuation

PR2707 merged into dev at `149acda36be8939fe8cd5e589bf77d13462257e7`.
The post-merge heartbeat is paused. Independent follow-ups remain draft:

- [PR2711](https://github.com/rmusser01/tldw_chatbook/pull/2711): TASK-32823
  inspector refresh ownership, 354 targeted cases and twelve native captures.
- [PR2712](https://github.com/rmusser01/tldw_chatbook/pull/2712): TASK-32825
  server-action ownership, 109 targeted cases and twelve native captures.
- [PR2713](https://github.com/rmusser01/tldw_chatbook/pull/2713): TASK-32829
  lifecycle cancellation, 35 targeted cases and twelve native captures.
- [PR2714](https://github.com/rmusser01/tldw_chatbook/pull/2714): TASK-32830
  connected catalog refresh/recovery, fifteen targeted cases and sixteen native
  captures. Applicable CI passes on `970f568c2f`; no submitted review at this checkpoint.
- TASK-32831: [tool error propagation](../qa/2026-09-18-mcp-tool-errors/README.md),
  eleven targeted cases and four inspected dark/light native captures at 170×48.
  Real stdio failures now reach Test Tool and Audit truthfully; successful retry
  uses the same connection. This branch does not include PR2711–PR2714 changes.

Every draft retains current-head CI, accumulated review and owner visual approval
as merge gates. A draft-skipped bot check is not substantive approval.

## Remaining review

- **Remote verification:** current-head checks and review belong to each follow-up
  PR. TASK-32814 repaired the earlier PR2707 rail ellipsis assertion failure.
- **Existing test debt:** four Audit CSS literal assertions fail against unchanged
  token-based CSS; two Speech harness cases failed during profile setup before UI
  creation. Both boundaries are recorded in TASK-32796 evidence. TASK-32812 also
  recorded 26 baseline CSS-consolidation ratchet offenders and a destination-tour
  profile-setup failure; TASK-32813 now closes those two boundaries with
  [computed-style, guard and native evidence](../qa/2026-09-18-css-consolidation/README.md).
- **Next: compact Test Tool reachability.** TASK-32831's 80×24 native probe cannot
  reveal its focused argument field, including after explicit scroll reveal.
  Retained captures and failed qualification receipts link from its QA report.
  Wide execution passes; compact execution remains unqualified.
- **Inspector refresh and execution:** argument drafts across background refresh
  (PR2711), diagnostic empty-state action routing, disconnected/stale tools,
  schema/raw arguments and inspector recovery. TASK-32831 qualifies local stdio
  failure/success outcomes, not the complete execution surface.
- **Servers:** source transitions, add/edit/remove and connection lifecycles,
  built-in enable/expose controls, errors and recovery. PR2714 qualifies local
  stdio catalog refresh; remote and full source transitions remain open.
  TASK-32822 repairs the
  compact non-master gate label clipping recorded in TASK-32793. The All servers
  rail clipping recorded in TASK-32796 is repaired by TASK-32812 above.
- **Audit and remaining permissions:** filtering, exact tool/event drilldown,
  restored roots, bulk actions, review/approval and connected-runtime journeys.

The [component completion ledger](2026-09-17-design-system-completion-audit.md)
retains other destinations. Follow-up drafts remain subject to their own visual
review and merge approval.
