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
| Selected-tool refresh, preview retirement and visible focus | TASK-32823, [354 distinct targeted cases and twelve native captures](https://github.com/rmusser01/tldw_chatbook/blob/0a30029bcf3c7715f9098a4e38d93926d6962e3e/Docs/superpowers/qa/2026-09-18-mcp-inspector-refresh/README.md) | Unchanged schema/raw drafts, changed/removed definitions, pending preview ownership, newer selection/focus and compact/wide control reachability; controlled catalog projection, no external server or tool execution |

| Server action identity and compact toolbar | TASK-32825, [109 distinct targeted cases and twelve native captures](https://github.com/rmusser01/tldw_chatbook/blob/2cd465c169582ff2a3c4bb0c6d18f4e16590176b/Docs/superpowers/qa/2026-09-18-mcp-server-actions/README.md) | Retired/hidden controls, accepted exact-target deletion, safe Keep/Escape and full compact/wide actions; real private deletion, no external connection/execution |

## Post-merge continuation

PR #2707 merged at `149acda36b` after owner approval, current-dev integration,
all ten Qodo findings resolved and eleven applicable Actions checks passed.
TASK-32823 resumed from that dev state on `codex/mcp-inspector-refresh-review`,
reusing preserved commit `135f226888`. The backup heartbeat is paused.

The bounded follow-up repairs stale selected details and pending preview
retirement, retains unchanged form/raw drafts, respects newer selection/focus,
and gives the inspector a scrollable viewport for keyboard-reached controls.
The bounded implementation is saved as [draft PR #2711](https://github.com/rmusser01/tldw_chatbook/pull/2711);
TASK-32823 is Done. Four final dark/light compact/wide native cells pass, with twelve inspected
captures and clean private-profile shutdown recorded in its receipt. Follow-up
PR2711’s selector-ratchet repair passed; all eight applicable Actions checks are green on `0a30029bcf`. Remote review and owner visual approval remain separate gates.
Server lifecycles, connected execution and the other remaining items stay open.


TASK-32825 continues on the independent branch `codex/mcp-server-action-ownership`
from merged dev. [109 distinct targeted checks and twelve inspected native captures](https://github.com/rmusser01/tldw_chatbook/blob/2cd465c169582ff2a3c4bb0c6d18f4e16590176b/Docs/superpowers/qa/2026-09-18-mcp-server-actions/README.md)
qualify retired/hidden server controls, immutable accepted deletion targets,
safe Keep/Escape and complete compact/wide toolbar paint. The native journey
removes only the confirmed Alpha profile while Beta stays selected and persisted.
All private lifecycle checks pass. The compact CSS rule uses existing action
classes to preserve the unchanged 274 selector ceiling. Broader server connection,
execution, recovery and other destination reviews remain open. The bounded repair is saved as [draft PR #2712](https://github.com/rmusser01/tldw_chatbook/pull/2712);
TASK-32825 is Done. All eight applicable Actions checks are green on `2cd465c169`; accumulated review and owner visual approval remain separate merge gates. Neither draft has substantive remote review; CodeRabbit skipped both drafts.

## Cancellation follow-up — TASK-32829

The independent `codex/mcp-lifecycle-cancellation-review` branch retains per-server
admission through actual worker settlement and final snapshot collection. Cancel
shows cleanup progress; repeated and retired controls cannot interrupt cleanup
or cancel a replacement. Lazy invocation also closes the before-start coroutine
leak. [Bounded evidence](../qa/2026-09-18-mcp-lifecycle-cancellation/README.md) records deterministic
regressions and native compact/wide dark/light checking, cancellation and retry.
The native client call uses controlled delays and failure; successful external
transport and connected tool execution remain unqualified. The follow-up retains
its own CI, remote review and owner visual approval gates.

## Remaining review

- **Remote verification:** All eleven applicable checks passed on PR2707 head
  `2ac5d7aab6` before merging. New follow-up PR checks qualify their own head.
- **Existing test debt:** four Audit CSS literal assertions fail against unchanged
  token-based CSS; two Speech harness cases failed during profile setup before UI
  creation. Both boundaries are recorded in TASK-32796 evidence. TASK-32812 also
  recorded 26 baseline CSS-consolidation ratchet offenders and a destination-tour
  profile-setup failure; TASK-32813 now closes those two boundaries with
  [computed-style, guard and native evidence](../qa/2026-09-18-css-consolidation/README.md).
- **Inspector execution and recovery:** TASK-32823 qualifies selected definition
  currentness, retained schema/raw drafts across unchanged catalog refresh and
  retiring form previews. Diagnostic empty-state action routing, disconnected/stale
  tools, argument validation, actual execution and post-execution recovery remain.
- **Servers:** TASK-32825 qualifies toolbar action ownership, safe confirmation and compact paint. Source transitions, complete add/edit/remove and connection lifecycles,
  built-in enable/expose controls, errors and recovery. TASK-32822 repairs the
  compact non-master gate label clipping recorded in TASK-32793. The All servers
  rail clipping recorded in TASK-32796 is repaired by TASK-32812 above.
- **Audit and remaining permissions:** filtering, exact tool/event drilldown,
  restored roots, bulk actions, review/approval and connected-runtime journeys.

The [component completion ledger](2026-09-17-design-system-completion-audit.md)
retains other destinations. PR2707 is merged; each follow-up PR retains its own visual review and merge gate.
