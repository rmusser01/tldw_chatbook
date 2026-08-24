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

The Items catalogue remains local-only with no backend selector or merged population. The existing
finished-server-ingest route may continue to open one explicitly labelled external server detail,
without inserting it into the local Items list. Server browse/search is outside this decision.

Complete stored text/Markdown remains authoritative. V1 rich preview is limited to capability-
gated local PNG, JPEG, and WebP originals rendered above—not instead of—the complete text. It never
fetches remote assets; other media types and server-item previews are outside this slice. Info
identifies provenance and the exact representation passed by Use in Console.

Local soft deletion includes a bounded Undo action implemented through the existing restore seam.
The one-off external server detail remains read-only. The redesign does not add unread or starred
state to Library media.

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
- Normal responsive resolution fits expanded panes at fixed target widths, collapses Library before
  Items on shortfall, and gives remaining columns to Reader. Only an explicit open may compress the
  requested pane toward its minimum after collapsing the other pane; this applies to both manually
  preferred-collapsed and responsively hidden panes.
- Settings persist only manual pane choices and optional normalized custom widths. Responsive
  collapse, focus, selection, pending loads, and active mode remain transient.
- Detail loading must distinguish selected from loaded ids and reject stale completions.
- Items filtering and pagination must use the existing scope-service contracts and stable backend-
  qualified identity.
- Existing media multi-select, bulk export, and bulk delete remain list-level capabilities; this
  design does not change their payload or partial-failure contracts and adds Undo only to the
  single-item delete flow.
- Local single-item delete UX must include Undo through the existing restore seam.
- Rich preview requires capability-on and capability-off behavior, with complete text always
  reachable.
- Watchlists and Library Media may duplicate a small amount of pane code initially. A later ADR can
  authorize extraction if the shipped contracts converge.
- Implementation plans and Backlog tasks must link this ADR and the approved design.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-23-library-media-netnewswire-reader-design.md)
- [Watchlists NetNewsWire reader design](../../Docs/superpowers/specs/2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md)
- [ADR-042: Watchlists reader-first information architecture](042-watchlists-reader-first-ia.md)
