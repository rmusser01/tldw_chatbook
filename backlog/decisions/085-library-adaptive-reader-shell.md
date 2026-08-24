# ADR-085: Share an adaptive reader shell within Library destinations

Status: Accepted
Date: 2026-08-24
Related Task: Implementation tasks will be created from the approved design during planning.

## Decision

Library Media, Conversations, Notes, Prompts, and Skills will share one Library-local adaptive
reader shell. The shell owns three structural regions, two full-height grips, effective pane
geometry, responsive collapse, focus-region integration, and late-bound content slots. It does not
own destination records, services, fields, modes, actions, drafts, conflicts, trust, imports, or
recovery behavior.

The Library rail and destination list are independently collapsible; the destination-owned work
pane is permanent. Requested visibility and normalized widths persist, while responsive overrides
and effective geometry remain transient. Library visibility is shared across the participating
destinations; list visibility and width are destination-specific. When Library is effectively
collapsed, reclaimed width expands the destination list toward its comfort cap before remaining
surplus flows to the work pane; it never shrinks a wider requested custom width that still fits.
Responsive resolution protects the active work mode, collapses Library before the list, and uses
stable thresholds plus hysteresis.

Shared Library geometry is persisted under `[library.reader]`; Media keeps its existing
`[library.media_reader]` list keys, and Conversations, Notes, Prompts, and Skills receive matching
destination-specific reader sections. The shared custom-width opt-in governs normalized shared and
destination widths. Existing Media Library values are a compatibility fallback until the shared
section is explicitly saved. Grips persist open/collapsed choices only; custom widths are changed
through Settings rather than by turning the grips into drag handles.

`LibraryScreen` remains the orchestration owner under the existing compose-once and scoped canvas
replacement contracts. Concrete destination list and work widgets remain destination-owned and
are supplied to the shell through late-binding builders or current state snapshots. The shared
layout policy is pure: it performs no database reads, destination loading, or preference writes.

The shell is extracted in the Conversations migration PR from the shipped Media reader. Media
retains its domain behavior and preference compatibility while adopting the shared structure. The
remaining migrations land in order: Notes, Prompts, then Skills.

This decision amends ADR-084's Media-local shell boundary now that multiple concrete Library
consumers have been designed. It does not authorize sharing an application-wide implementation
with Watchlists.

## Context

ADR-084 deliberately kept the first adaptive reader inside Media and deferred extraction until
concrete consumers existed. Conversations, Notes, Prompts, and Skills now have approved reader
contracts with the same structural roles, collapse grammar, requested-versus-effective geometry,
and race-safe selected/loaded identity. Keeping four additional copies would multiply subtle
resize, focus, preference, and worker-race behavior.

The destinations still differ substantially in their domain behavior. Conversations is a
read-only transcript, Notes and Prompts own different draft/conflict models, and Skills has a trust
boundary plus supporting files. A generic data-driven workbench would erase those boundaries and
make capability preservation harder to verify.

This is a durable Library application-structure and cross-module interface decision, so it is
recorded as an ADR.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Clone the Media reader into each destination | It duplicates difficult geometry, focus, persistence, and stale-worker behavior across five implementations. |
| Build an application-wide split-pane framework shared with Watchlists | Library and Watchlists still have different lifecycle, ownership, and pane contracts; evidence does not justify the broader boundary. |
| Build a schema-driven generic editor/action framework | Conversations, Notes, Prompts, and Skills have materially different drafts, validation, conflicts, trust, and workflows. |
| Keep list-to-work takeovers | It breaks scan/select/work continuity and prevents the destination list from remaining available during secondary workflows. |
| Persist effective responsive geometry | Terminal resizing would overwrite deliberate choices and make panes fail to restore predictably. |
| Ship all four migrations in one PR | It creates an unsafe cutover, obscures capability regressions, and prevents destination-by-destination verification. |
| Add a separate foundation-only PR | The extraction has no independent user value; landing it with Conversations keeps the programme to four releasable PRs. |

## Consequences

- One Library-local structural shell and pure geometry policy become shared code.
- `LibraryScreen` remains the router and controller composition owner.
- Each destination keeps concrete list/work widgets and its current domain services.
- Media gains the approved list-comfort expansion while preserving its other behavior and existing
  preference values through compatibility normalization.
- Library visibility and custom-width opt-in have one shared preference owner; each destination
  list has its own visibility and width keys.
- Responsive adaptation never performs data work or writes preferences.
- Every detail load distinguishes selected and loaded identity and rejects late results using a
  complete destination/item/revision/generation fence.
- Capability inventories and destination regression tests are required for each migration.
- The programme lands as Conversations, Notes, Prompts, and Skills PRs in that order.
- Watchlists sharing remains a future decision requiring separate evidence and an ADR.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md)
- [ADR-084: Library Media reader information architecture](084-library-media-reader-ia.md)
- [Library Media reader design](../../Docs/superpowers/specs/2026-08-23-library-media-netnewswire-reader-design.md)
- [Library compose-once design](../../Docs/superpowers/specs/2026-08-13-library-compose-once-design.md)
