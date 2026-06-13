# Library Collections IA Split Design

Date: 2026-05-08
Status: User-approved design direction; implementation plan drafted
Branch baseline: `origin/dev` at `df63b49f` (`Close Gate 1.6 Library Search/RAG (#281)`)

## Related Documents

- `Docs/superpowers/plans/2026-05-06-gate-1-core-product-loop-screen-adaptation.md`
- `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`
- `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `Docs/superpowers/plans/2026-05-08-phase-3-9-library-collections-ia-split.md`

## Summary

Phase 3.9 splits the current combined Watchlists+Collections product model into two clearer destinations:

- **Watchlists** remains a top-level destination for monitored sources, active runs, alerts, schedules, and Console follow-through.
- **Collections** moves into **Library** as a Library-owned mode for grouping saved Library sources for later Search/RAG, Study, Console, and future artifact workflows.

The first implementation slice is local-first and management-capable. Users should be able to discover Collections from Library, create a named collection, select it, rename it, and delete it. Server parity remains a future goal because the sync engine is not written yet. The UI must not expose working-looking sync controls until sync actually exists.

## Problem Statement

The current `W+C` model conflates two different user intents:

- Watchlists answer: "What external sources am I monitoring and what runs need attention?"
- Collections answer: "How do I organize saved Library sources into reusable sets?"

That conflation weakens first-time comprehension and power-user routing. It also creates a risk that future Collection work gets implemented inside Watchlist services or screen copy, where it will be harder to integrate with Library Search/RAG, Study, Console, and artifacts.

Library already exposes a `Collections` mode, but it is currently placeholder-level. Phase 3.9 should make that mode real enough to validate the product model without overbuilding server sync or cross-module source pickers.

## Goals

- Rename the visible top-level `W+C` destination to **Watchlists**.
- Keep existing route IDs or aliases where needed for compatibility.
- Remove user-facing Collections copy from the Watchlists destination.
- Add a Library-owned Collections mode that supports local create, select, rename, and delete.
- Represent Collections as reusable Library source sets, not watchlist runs.
- Preserve power-user density and keyboard-friendly navigation.
- Make sync status honest: local-only until a sync engine exists.
- Add tests and QA evidence proving the app is usable, not merely renderable.
- Keep future citations and snippets visible in the roadmap as later-stage Library/Search/RAG capabilities.

## Non-Goals

- No server sync engine implementation.
- No server parity beyond honest local-only metadata and future seams.
- No full cross-source item picker unless an existing Library service makes it safe and small.
- No Collection-scoped Search/RAG execution in this slice.
- No Collection-scoped Study/Flashcards/Quizzes generation in this slice.
- No Import/Export changes for Collections in this slice.
- No Watchlists runtime rewrite.
- No route-ID breaking migration.

## Product Model Contract

### Navigation

Visible primary navigation should use these concepts:

- `Home`
- `Console`
- `Library`
- `Personas`
- `Watchlists`
- `Schedules`
- `Workflows`
- `MCP`
- `ACP`
- `Skills`
- `Artifacts`
- `Settings`

`Collections` is not a top-level destination. It is a Library mode.

### Compatibility

Existing internal names such as `watchlists_collections` may remain as compatibility aliases for the Watchlists destination during this gate. User-facing labels, command palette copy, help text, breadcrumbs, and empty states should say **Watchlists**, not `W+C` or `Watchlists+Collections`.

Home, Console, and active-work surfaces should also use **Watchlists** as the user-facing source label for monitored-source runs. Internal payload values may remain compatibility-oriented where changing them would break existing routing, but visible copy and new tests should not teach users that `W+C` is the product model.

### Library Mode Placement

Library should expose Collections alongside its other Library modes. Collections belongs with source organization because it will eventually feed Search/RAG, Study, Console, citations/snippets, and artifacts.

## Library Collections Behavior

### First Slice Capabilities

The first implementation slice must support:

- List local collections.
- Create a collection with a required name and optional description.
- Select a collection.
- Rename a selected collection.
- Delete a selected collection with a clear confirmation or recovery affordance.
- Show item count, updated timestamp, and local-only sync status.
- Show an empty state that explains why Collections matter.

Suggested empty-state copy:

> Group saved Library items for Search/RAG, Study, and Console.

### Collection Fields

The display contract should support:

- `collection_id`
- `name`
- `description`
- `item_count`
- `updated_at`
- `source_authority`
- `sync_status`

For this gate:

- `source_authority` is `local`.
- `sync_status` is `local-only` or `sync-unavailable`.

### Item Membership

This phase should not block on a full item picker. If existing Library services make adding the current selected item small and safe, it may be included. Otherwise, item membership should be represented as an explicit follow-up seam with honest empty/detail copy.

The UI must not imply that Collection-scoped RAG, Study, or Console actions are already functional unless those flows are implemented and verified.

## Watchlists Split Behavior

The existing combined destination should become Watchlists-only.

Watchlists should retain:

- Monitored source overview.
- Watchlist run status.
- Active run follow-through from Home and Console.
- Alert or notification orientation.
- Schedule/run recovery surfaces if currently present.

Watchlists should remove:

- User-facing Collections title text.
- Collection management controls.
- Collection count summaries.
- Copy that describes Collections as part of the Watchlists destination.

If a compact navigation label is required, `Watchlists` is still preferred. Avoid `W+C` because Collections is no longer part of that destination.

## Service Boundaries

Add or adapt a Library-owned service boundary rather than reusing Watchlist services as the Collection model.

Recommended contract:

```text
LibraryCollectionsService
  list_collections() -> list[LibraryCollection]
  get_collection(collection_id) -> LibraryCollection | None
  create_collection(name, description="") -> LibraryCollection
  rename_collection(collection_id, name, description=None) -> LibraryCollection
  delete_collection(collection_id) -> None
```

Recommended display-state models:

```text
LibraryCollectionsPanelState
  status: loading | ready | empty | error
  collections: list[LibraryCollectionSummary]
  selected_collection: LibraryCollectionDetail | None
  actions: LibraryCollectionActions
  error_message: str | None
```

The Library screen should depend on the service and display-state contract, not directly on Watchlist internals.

Existing Watchlist, read-it-later, feed, and media-reading services may later become adapters into Collections. They should not define the first Library Collection contract.

## Textual Layout Contract

Collections should fit the existing Library shell rather than introduce a new visual language.

```text
+- Library --------------------------------------------------------------+
| Modes: Sources  Search/RAG  Collections  Import/Export                 |
+--------------------+-------------------------------+------------------+
| Collection List    | Selected Collection           | Inspector        |
|                    |                               |                  |
| + New Collection   | Name                          | Actions          |
|                    | Description                   | Rename           |
| Recent Collections |                               | Delete           |
|                    | Items                         |                  |
| Empty/Error state  | Empty membership/future seam  | Sync: local-only |
+--------------------+-------------------------------+------------------+
```

This is a structural contract, not a pixel-level mockup. The implementation should reuse existing Library layout, tokens, buttons, and shell patterns.

## Error And Recovery Contract

Collections errors should be specific and recoverable:

- Invalid name: keep user input visible and explain the validation rule.
- Create/rename/delete failure: keep current list visible where possible and show retry.
- Missing storage/service: show `Collections unavailable` with local ownership and next step.
- Sync unavailable: show status only; do not render a fake sync action.

Destructive delete should either require confirmation or provide an immediate recovery path appropriate for the existing Textual patterns.

## Future Roadmap Hooks

These are explicitly not part of the first Phase 3.9 implementation, but should remain visible in roadmap/spec follow-ups:

- Collection item membership across Library media, notes, ingested files, imports, and external feeds.
- Collection-scoped Search/RAG from Library.
- Collection-scoped Chat/Console RAG.
- Collection-scoped Study generation for flashcards and quizzes.
- Citations and snippets in Search/RAG answers.
- Citation/snippet carry-through into Chat, artifacts, and exported Chatbooks.
- Server sync once a sync engine exists.
- Import/Export of collection definitions and membership.

## Testing Contract

Implementation should be test-driven.

Required regression coverage:

- Pure service/display-state tests for list, create, rename, delete, empty, and error states.
- Mounted Library tests proving Collections mode renders and basic management controls work.
- Mounted Watchlists tests proving the destination no longer presents Collections as part of the top-level product model.
- Navigation or command-palette tests proving the visible destination is Watchlists and Collections is discoverable under Library.
- Existing destination-shell and Gate 1.6 Library Search/RAG tests remain green.

Recommended focused verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_product_maturity_phase3_layout_contracts.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_library_search_rag_screen.py \
  --tb=short

git diff --check
```

Exact test files may change with implementation scope, but verification must cover both the Watchlists split and Library Collections management.

## QA Walkthrough Gate

This phase is not done when widgets render. It is done when a manual QA walkthrough confirms the flows are understandable and usable.

Required QA checks:

- First-time user can find Collections from Library without knowing the old `W+C` label.
- User can create, select, rename, and delete a local collection.
- Empty states explain what Collections are for and what is not available yet.
- Watchlists still opens and explains monitored-source workflows.
- Watchlist active-run follow-through from Home/Console still works if covered by existing fixtures.
- No fake server sync action appears.
- No visual breakage in the Library three-region shell at normal terminal sizes.

QA evidence should be recorded in the product-maturity QA docs or tracker with commands, screenshots/snapshots if helpful, observed failures, and residual risks.

## Acceptance Criteria

- Watchlists is visibly split from Collections in navigation, command palette/help copy, and destination body copy.
- Collections is discoverable inside Library and has local management affordances.
- Local create, select, rename, and delete are covered by focused tests.
- Library Collections display state exposes local-only/sync-unavailable status honestly.
- Existing Watchlists workflows are preserved.
- Existing Library Search/RAG gate tests remain green.
- Roadmap or follow-up notes include citations/snippets as later-stage Library/Search/RAG work.
- QA walkthrough evidence confirms the app is usable for the targeted flows.

## Open Questions For Implementation Planning

- Which existing local persistence layer should own the first Collection records: a new DB table, an existing Library DB boundary, or a small adapter around current media/reading services?
- Should delete be confirm-first or undo-capable based on existing Textual patterns?
- Should first-slice item membership include "add current selected Library item" if a selected item contract already exists?
- Which route aliases are required for backwards-compatible tests and saved navigation state?
