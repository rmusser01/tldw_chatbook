# ADR-104: Settle Library Media returns at current-owner geometry

Status: Accepted
Date: 2026-08-30
Related Task: TASK-18918
Amends: [ADR-084](084-library-media-reader-ia.md), [ADR-086](086-library-adaptive-reader-shell.md)

## Decision

Library Media will restore retained focus and physical list scroll through a two-gate,
event-driven settlement boundary. The Media compose branch projects the correct adaptive
presentation from its first frame, and the current Media adaptive shell equality-
reconciles that projection after mount. The current Media row-scroll owner then reports
its own Resize-derived geometry after Textual reflow. Only an authority-fenced request
whose exact offset is representable may publish exact settlement.

`LibraryScreen` owns request, route, generation, content/layout revision, timeout, final
focus policy, and the association between geometry revision and presentation epoch. A
small Media-owned scroll widget reports owner identity and public Resize-derived
geometry but owns no navigation, epoch, or domain state. Real presentation changes
advance an application epoch and establish an exclusive floor over already-emitted
owner geometry; real later owner geometry events trigger settlement. Viewer
returns finish on the Media row, while Trash round trips finish on their captured normal-
Media control after the retained list scroll is exact.
Fixed scheduler-turn counts, arbitrary sleeps, recursive polling, and framework-private
layout signals are not readiness contracts.

Exact physical scroll remains valid only while the captured content and viewport/layout
revisions remain unchanged. A proven revision may produce one honest semantic return
with clamping and a distinct outcome. Missing eligible geometry reaches one bounded,
privacy-safe failure at the existing two-second focus deadline and does not requeue.

This decision is Media-specific. It does not generalize the protocol across Notes,
Conversations, Prompts, Skills, Watchlists, or application-wide split panes without new
evidence and a later decision.

## Context

TASK-18918's cross-reader gate showed that a correct normal-Media return receipt could
restore `(0, 33)` instead of `(0, 42)` after authoritative recompose. Investigation
proved that replacement shells could retain the legacy Notes compact class after a
same-size recompose, leaving the owner under the wrong layout contract. It also proved
that one or more later callbacks are not descendant-layout completion signals in
Textual 8.2.8.

Textual's public widget Resize boundary is emitted after it installs size, virtual size,
and container size, updates scrollbars, and clamps scroll. That event is sufficient only
after the application has reconciled the current Media presentation. This creates a
durable cross-module interface and amends the adaptive shell lifecycle, so an ADR is
required rather than treating the work as routine UI polish.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Restore immediately after compose | Silently clamps against transient or wrong-contract geometry. |
| Reconcile presentation and retain immediate restore | Corrects one producer defect but leaves the Textual layout race. |
| Delay by N scheduler turns | Four progressively broader trials remained nondeterministic in fresh processes. |
| Poll scroll maxima | Uses elapsed execution as authority, can loop, and cannot distinguish legitimate content shrink. |
| Subscribe to private screen layout signals | Couples application behavior to non-public Textual internals without solving application-epoch authority. |
| Generalize a shared Library settlement framework now | Only Media has complete failure evidence and an approved exact-return contract; broader extraction would be speculative. |

## Consequences

- Same-size Media recomposes must compose the correct adaptive presentation initially
  and reconcile it after mount.
- Exact return success becomes an explicit event-driven state instead of an incidental
  result of callback order.
- Media gains a small owner-geometry message seam and transient settlement state.
- Stale owners, epochs, requests, routes, content, layouts, and unmounted trees fail
  closed.
- A rare missing-geometry path may report one bounded fallback rather than falsely
  claiming exact restoration.
- Tests must prove deterministic fresh-process behavior and cross-reader isolation.
- Other Library destinations remain unchanged.

## Rollback plan

Remove the Media settlement request/message seam and post-mount projection hook while
leaving the retained receipt format unchanged. The previous immediate focus/scroll path
can be restored mechanically, accepting its documented nondeterministic clamp. No
stored data or migration is involved.

## Links

- [Library Media return settlement design](../../Docs/superpowers/specs/2026-08-30-library-media-return-settlement-design.md)
- [ADR-084: Library Media reader information architecture](084-library-media-reader-ia.md)
- [ADR-086: Shared adaptive Library reader shell](086-library-adaptive-reader-shell.md)
