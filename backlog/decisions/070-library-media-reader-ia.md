# ADR-070: Make Library Media an adaptive reader with a permanent Reader

Status: Accepted
Date: 2026-08-23
Related Task: Implementation tasks will be created from the approved design during planning.

## Decision

The Library Media destination will use a three-region, NetNewsWire-shaped information architecture:
an independently collapsible Library rail, an independently collapsible Items list, and a permanent
Reader. Reader is the spatial and width priority and is never a collapse target.

The layout distinguishes persisted **preferred** pane state from temporary **responsive** overrides
and derives the rendered **effective** state from both. Resolution uses the width available to the
Media shell and declared pane minimums, collapses Library before Items, retains a five-column grip
for each pane, and uses hysteresis to avoid resize thrash. Fixed target widths are the default;
custom widths are opt-in and persisted only after normalization.

Library Media and Watchlists share the same user-facing reader grammar—permanent centre content,
full-height five-column ASCII grips, and preferred-versus-responsive state—but remain independent
implementations. No application-wide split-pane abstraction will be introduced until both
consumers have shipped and their genuinely common contract can be extracted.

The current media list/viewer state and `media_reading_scope_service` remain authoritative.
Selection and loaded Reader identity are separate, and every detail request is guarded by backend-
qualified id plus a request generation. The Items list uses the existing paginated list and search
contracts rather than filtering only its initially loaded records.

Complete stored text/Markdown remains the authoritative readable fallback. Rich media preview is
optional, capability-gated enhancement and must never block access to the complete stored
representation. Info identifies provenance and the exact representation passed by Use in Console.

Soft deletion includes a bounded Undo action implemented through the existing restore seam. The
redesign does not add unread or starred state to Library media.

## Context

The current Library Media flow replaces the item list with an in-canvas viewer. Although the viewer
already has strong item capabilities, moving between items breaks the scan/read rhythm and removes
the list context. The desired NetNewsWire model keeps navigation, contextual items, and readable
content visible together while allowing side panes to yield width on smaller terminals.

Watchlists is adopting a similar reader grammar in parallel. Sharing interaction language makes the
application coherent, but the implementations currently differ in pane count, ownership, backend
semantics, and lifecycle. Premature extraction would couple two active redesigns to assumptions not
yet tested in either consumer.

This is a long-lived application-structure and preference decision, so it is recorded as an ADR
rather than only as routine UI polish.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep the current list-to-viewer takeover | It hides traversal context and cannot deliver a continuous scan/read workflow. |
| Add a separate Media reader screen | It duplicates Library navigation and authoritative media/viewer state. |
| Share a split-pane framework with Watchlists immediately | The common implementation contract is not established; sharing grammar now and extracting after both ship avoids speculative coupling. |
| Make Reader collapsible | Reader is the purpose and stable spatial anchor of the destination. |
| Persist the rendered effective layout | Responsive collapses would overwrite deliberate user choices and make resize behavior surprising. |
| Filter only the first loaded media page | Results would be incomplete and misleading despite existing backend search/pagination seams. |
| Require rich rendering | Textual/headless/runtime capabilities vary; complete stored text is the dependable authoritative representation. |

## Consequences

- Media gains a destination-local shell and pure session/layout resolver while `LibraryScreen`
  remains the orchestration owner.
- Both collapsed panes continue to consume five columns each for reachable grips; Reader receives
  all remaining width.
- Settings persist only manual pane choices and optional normalized custom widths. Responsive
  collapse, focus, selection, pending loads, and active mode remain transient.
- Detail loading must distinguish selected from loaded ids and reject stale completions.
- Items filtering and pagination must use the existing scope-service contracts and stable backend-
  qualified identity.
- Delete UX must include Undo because both local and server restore seams already exist.
- Rich preview requires capability-on and capability-off behavior, with complete text always
  reachable.
- Watchlists and Library Media may duplicate a small amount of pane code initially. A later ADR can
  authorize extraction if the shipped contracts converge.
- Implementation plans and Backlog tasks must link this ADR and the approved design.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-23-library-media-netnewswire-reader-design.md)
- [Watchlists NetNewsWire reader design](../../Docs/superpowers/specs/2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md)
- [ADR-042: Watchlists reader-first information architecture](042-watchlists-reader-first-ia.md)
