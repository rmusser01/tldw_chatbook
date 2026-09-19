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

## Post-merge follow-ups

PR [2707](https://github.com/rmusser01/tldw_chatbook/pull/2707) merged into dev at
149acda36be8939fe8cd5e589bf77d13462257e7. Its post-merge heartbeat is paused;
component review resumed on separate bounded branches. Earlier closeout notes
are historical, not a current merge gate for that already-merged PR.

The following independently based draft PRs retain their own evidence and visual
approval requirement: [2711 inspector refresh](https://github.com/rmusser01/tldw_chatbook/pull/2711),
[2712 server action ownership](https://github.com/rmusser01/tldw_chatbook/pull/2712),
[2713 lifecycle cancellation](https://github.com/rmusser01/tldw_chatbook/pull/2713),
[2714 connected catalog refresh](https://github.com/rmusser01/tldw_chatbook/pull/2714),
and [2716 tool-error propagation](https://github.com/rmusser01/tldw_chatbook/pull/2716).
PR2716's applicable remote checks pass on 5d3e1cb97b; these changes are not included
in the inspector reachability branch.

TASK-32832, `codex/mcp-inspector-scroll-review`, repairs the compact Test Tool
viewport, action wrapping and focused-control reveal after resize. Its
[42 distinct targeted checks and 12 inspected native captures](../qa/2026-09-18-mcp-inspector-reachability/README.md)
qualify schema/raw argument draft retention, Ask execution through a real private
stdio process, validation recovery, Close/reopen and sibling button geometry.
Seven preflight guards and independent review pass. Six older selected tests
have matching profile-setup errors on unchanged and modified source; they are
not counted as passing. This bounded qualification does not complete MCP review.

TASK-32833, `codex/mcp-inspector-result-review`, builds on PR2718's reachability
fix. Its [50 passing targeted cases and 16 inspected native captures](../qa/2026-09-18-mcp-inspector-results/README.md)
qualify the compact Raw response disclosure/body, replacement of old raw/note
content on local validation failure, and corrected execution in the same real
private stdio session. Default/named profile contexts and raw/schema draft
retention are covered. Seven preflight guards, source/runner hashes and native
lifecycle pass. This follow-up is stacked on PR2718; retarget dev after that
parent merges and recheck integration before visual approval and merge.

## Remaining review

- **Remote verification:** each follow-up PR requires checks on its own pushed head.
- **Existing test debt:** four Audit CSS literal assertions fail against unchanged
  token-based CSS; two Speech harness cases failed during profile setup before UI
  creation. Both boundaries are recorded in TASK-32796 evidence. TASK-32812 also
  recorded 26 baseline CSS-consolidation ratchet offenders and a destination-tour
  profile-setup failure; TASK-32813 now closes those two boundaries with
  [computed-style, guard and native evidence](../qa/2026-09-18-css-consolidation/README.md).
- **Inspector refresh and execution:** selected tool definition/currentness,
  argument drafts across background refresh, diagnostic empty-state action routing,
  disconnected/stale tools, schema/raw arguments, execution and inspector recovery.
  TASK-32833 closes the compact Raw response title/content and local-validation
  replacement gaps recorded by TASK-32832. Broader schema and execution/preview
  ownership states still need their own qualification. Next destination slice:
  Audit filtering and exact event/tool drilldown.
- **Servers:** source transitions, add/edit/remove and connection lifecycles,
  built-in enable/expose controls, errors and recovery. TASK-32822 repairs the
  compact non-master gate label clipping recorded in TASK-32793. The All servers
  rail clipping recorded in TASK-32796 is repaired by TASK-32812 above.
- **Audit and remaining permissions:** filtering, exact tool/event drilldown,
  restored roots, bulk actions, review/approval and connected-runtime journeys.

The [component completion ledger](2026-09-17-design-system-completion-audit.md)
retains other destinations. Follow-up drafts remain subject to their own visual
review and merge approval.
