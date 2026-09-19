# MCP component review

This ledger separates qualified repairs from remaining destination behavior.
Token migration alone does not qualify a complete workflow.

## Post-merge continuation: Console approval actions

PR2707 has merged; its closeout statements below are historical checkpoints.
The continuation heartbeat is paused. TASK-32841 is an independent bounded
follow-up from merged dev, separate from draft PR2727/PR2728.

Queued Console Approve all, Deny all and Submit actions now retain their displayed
batch identity, and each batch can submit only once. Unchanged resyncs preserve
choices; changed/cleared/finishing batches invalidate old actions and fresh rounds
remain usable. [123 passing targeted cases, 19 identical baseline failures and
eight inspected native captures](../qa/2026-09-19-approval-action-ownership/README.md)
cover the existing controls. All seven derived guards pass, including a separate
Canvas input-fetch retry. Native uses synthetic pending calls with the real
controller and never dispatches a tool. No connected-runtime claim is made.

The proposed MCP permission-matrix bulk-actions ADR is not implemented here.
Next: session-grant review/revocation and remaining approval/connected-runtime
journeys. This draft requires current-head CI/review and final visual approval
before merge; the wider component review remains open.

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

## PR #2707 closeout boundary

The owner approved moving remaining MCP reviews to follow-up PRs. TASK-32823
is preserved on `codex/mcp-inspector-refresh-followup` at `135f226888`, outside
PR #2707. Its inspector refresh prototype has an unresolved delayed-preview
ownership race; it is not qualified or complete. Resume MCP lifecycle/inspector
work from merged `dev` after the current PR receives final visual approval and
its merge is confirmed.

## Remaining review

- **Remote verification:** Fast Lane, CSS, latency, backlog and all platform GGUF
  checks pass on saved head d7b4c10a7d. The derived-artifact aggregate is still
  running at this checkpoint; checks on the next pushed head are separate.
  TASK-32814 repaired the earlier rail ellipsis assertion failure.
- **Existing test debt:** four Audit CSS literal assertions fail against unchanged
  token-based CSS; two Speech harness cases failed during profile setup before UI
  creation. Both boundaries are recorded in TASK-32796 evidence. TASK-32812 also
  recorded 26 baseline CSS-consolidation ratchet offenders and a destination-tour
  profile-setup failure; TASK-32813 now closes those two boundaries with
  [computed-style, guard and native evidence](../qa/2026-09-18-css-consolidation/README.md).
- **Inspector refresh and execution:** selected tool definition/currentness,
  argument drafts across background refresh, diagnostic empty-state action routing,
  disconnected/stale tools, schema/raw arguments, execution and inspector recovery.
- **Servers:** source transitions, add/edit/remove and connection lifecycles,
  built-in enable/expose controls, errors and recovery. TASK-32822 repairs the
  compact non-master gate label clipping recorded in TASK-32793. The All servers
  rail clipping recorded in TASK-32796 is repaired by TASK-32812 above.
- **Audit and remaining permissions:** filtering, exact tool/event drilldown,
  restored roots, bulk actions, review/approval and connected-runtime journeys.

The [component completion ledger](2026-09-17-design-system-completion-audit.md)
retains other destinations. Draft PR2707 remains open and subject to its own
visual review and merge approval.
