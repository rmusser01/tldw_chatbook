# MCP component review

This ledger separates qualified repairs from remaining destination behavior.
Token migration alone does not qualify a complete workflow.

## Current closeout checkpoint

PR2707 is merged and its heartbeat remains paused. The owner approved the
PR2731/PR2734 continuation and visual evidence. Both follow-ups are ready, with
Qodo's accumulated findings resolved; runner validation and documentation fixes
preserve the approved application behavior. Both now target dev for required CI,
with PR2731 explicitly required to merge before PR2734. Historical draft/pending
approval statements below describe earlier checkpoints, not current status.
The next bounded review is remaining permission/Audit navigation ownership,
followed by connected-runtime journeys; neither is part of these PRs.

## Post-merge continuation: session-grant revocation

PR2707 is merged; its open-PR statements below are historical. The continuation
heartbeat remains paused. TASK-32865 is an independent bounded follow-up from
dev, separate from drafts PR2727/2728/2730.

Revoke binds to its mounted button's grant and profile, invalidates replaced
controls before removal, and admits each control once. Failed revocation can be
retried; late completion preserves a newer profile's listing.
[26 passing targeted checks and eight inspected native captures](../qa/2026-09-19-mcp-session-revocation/README.md)
cover the repair. All seven derived guards pass. Native keyboard revocation uses
the real service and runtime gate: only the selected grant clears, another tool
and profile retain their grants, and calculator asks again. No tool executes.

The follow-up remains draft pending current-head CI/review and owner visual
approval. Exact-input rule removal, Re-allow and other permission actions remain
next; connected external runtime journeys and the wider destination review stay
open.

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

## Post-merge continuation: exact-input Remove and Re-allow

TASK-32866 follows PR2731 on a separate stacked branch, `codex/mcp-rule-action-review`.
Both controls now capture their mounted target and submit once; obsolete controls
are rejected. Re-allow retains the reviewed definition fingerprint across cached
refreshes, retry and successful completion. Late completion preserves newer
selections, and failed writes remain retryable. Existing permission policy and
layout are preserved.

[56 passing targeted checks and twelve inspected native captures](../qa/2026-09-19-mcp-rule-actions/README.md)
qualify this bounded repair in both themes at 120×40 and 170×48. Real private
store/service writes remove only the selected rule and persist the reviewed
definition; another profile's rule survives. No connection or tool execution.
Seven preflight guards, unchanged static-analysis baseline, independent review,
normal shutdown, lock release, database health and source hashes pass.
Current-head CI/review and owner visual approval remain merge gates. Other
permission controls, connected runtime journeys and the wider review remain open.

## Tool permission navigation continuation — TASK-32867

PR2731 (session Revoke) and PR2734 (exact-input Remove/Re-allow) are now merged
into dev, at `7a695b5e73` and `6095db2f5f` respectively. This continuation starts
from that merged state on `codex/mcp-navigation-action-review`.

Both Tools inspector and Test Tool “Change in Permissions” controls now retain
the displayed tool/profile and reject retired, hidden, disabled or covered-screen
presses. Live navigation remains retryable. [75 targeted passing cases, independent
review and twelve inspected native captures](../qa/2026-09-19-mcp-permission-navigation/README.md)
verify the repair. Eight native keyboard routes select the expected permission row
without changing its record; no server connects or tool executes. All seven
preflight guards and private lifecycle checks pass.

Audit navigation is already saved in PR2724 / TASK-32837; it was not duplicated.
That existing draft currently conflicts with dev and still requires integration
review. Other saved MCP drafts, connected-runtime journeys and remaining screen
reviews stay open. This new repair is a separate draft against dev and requires
current-head CI/review plus its own final owner visual approval before merge.
