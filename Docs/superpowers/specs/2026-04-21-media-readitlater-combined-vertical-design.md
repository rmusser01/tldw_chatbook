# tldw_chatbook Media And Read-it-Later Combined Vertical Design

**Date:** 2026-04-21  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next standalone-first parity vertical after runtime-policy tranche 0: extend the existing `Media / Reading / Ingestion Sources` seam so `tldw_chatbook` can support a source-scoped `Read-it-later` experience inside `Media` while also exposing the current server ingestion-source management contract inside `media_ingest_screen`.

This vertical is intentionally combined. It does not treat `Read-it-later` as a separate destination or separate storage system. It treats it as a saved-state filter over the same source-scoped media and reading records that the user is already browsing.

## Context

This vertical is not starting from zero.

Recent work already established a real media/reading seam in `tldw_chatbook`:

- `tldw_chatbook/Media/media_reading_scope_service.py`
- `tldw_chatbook/Media/server_media_reading_service.py`
- `tldw_chatbook/Media/local_media_reading_service.py`
- `tldw_chatbook/Media/media_reading_normalizers.py`
- `tldw_chatbook/UI/Screens/media_screen.py`
- `tldw_chatbook/UI/Screens/media_runtime_state.py`
- `tldw_chatbook/UI/MediaWindow_v2.py`
- `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
- `tldw_chatbook/DB/Client_Media_DB_v2.py`

That means the current design problem is not "how do we invent a media seam?" The design problem is:

- how to extend that seam for `Read-it-later`
- how to keep local and server write authority explicit
- how to use the current server contract honestly instead of overclaiming it
- how to keep the resulting UX inside the existing `Media` and `Ingest` shells rather than creating a second browsing product

On the server side, the relevant contract is still split across explicit route families:

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/items.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`

The current server contract matters in two important ways:

1. `Read-it-later` is not represented as a native boolean field. The compatible server-side meaning is a reading/item lifecycle state, especially `status="saved"`.
2. Ingestion-source management already supports create, list, detail, update, item listing, sync trigger, and archive upload. It does not yet expose a delete route that this vertical can rely on.

This design therefore has to be contract-first and source-honest.

## Product Decisions

The following decisions are fixed for this vertical:

- This is one combined vertical:
  - `Media / Reading / Ingestion Sources`
  - `Collections: Read-it-later`
- `Read-it-later` stays inside the existing `Media` destination.
- `Read-it-later` is not a top-level destination.
- `Read-it-later` contains only real media/reading records, not lightweight saved-link stubs.
- `Read-it-later` is implemented as a saved-state/filter model, not as a distinct collection entity with explicit membership records.
- Local and server views remain source-scoped and source-labeled.
- No mixed local/server list is introduced.
- Local mode writes only to local state.
- Server mode writes only to server state.
- No silent dual-write, sync, mirror, or reconciliation behavior is introduced.
- In `server mode`, `Read-it-later` is a compatibility mapping:
  - save => `status="saved"`
  - remove => `status="archived"`
- In `server mode`, `Read-it-later` is first-class in the aggregate `All Media` browse surface for this slice.
- Per-media-type server `Read-it-later` subfilters are deferred until the server browse contract can support them without client-side drift.
- `favorite` remains a separate concern and is not reused as `Read-it-later`.
- `read_it_later_saved_at` is an optional compatibility field and is local-authoritative in this slice.
- `Media` remains the primary browsing surface.
- `media_ingest_screen` remains the ingestion/source-management surface.
- In `server mode`, the ingestion surface targets the current server contract that exists now:
  - create
  - list
  - detail
  - update
  - list items
  - trigger sync
  - archive upload
- First-slice server ingestion-source create is intentionally narrowed to remote-client-sensible source types:
  - `archive_snapshot`
  - `git_repository`
- `local_directory` creation is deferred as a server-local concern until a better remote-client UX exists.
- Server-side ingestion-source delete is explicitly deferred unless the server adds it.
- Runtime policy remains the authority for source-aware permission and offline behavior.
- This slice may extend the current media seam and window boundaries, but it should not become a broad Media UX redesign.

## In Scope

- Extend the normalized media contract with `Read-it-later` compatibility fields.
- Add local saved-state persistence for `Read-it-later`, keyed by local media ID.
- Add backend-native filtering for the `Read-it-later` subview:
  - local mode via local persistence-aware queries
  - server mode via `status=saved` request shaping
- Add source-aware `Read-it-later` mutation through the scope service.
- Add explicit UI-facing scope-service operations for `Read-it-later` list/save/remove behavior.
- Extend the current server media-reading service and scope service to support ingestion-source create in addition to the already landed list/detail/update/items/sync/archive operations.
- Refine `Media` browsing so `Read-it-later` is a first-class subview/filter over the same source-scoped records.
- Refine `media_ingest_screen` so server mode exposes the full currently-supported ingestion-source management contract.
- Preserve source-aware reading-progress behavior for ordinary and saved-for-later records.
- Add explicit lifecycle rules for local saved-state on trash/delete.
- Add regression coverage for:
  - normalized saved-state fields
  - local persistence
  - source-native filter behavior
  - server compatibility mapping
  - ingestion-source create plus existing management flows
  - backend switch invalidation
  - UI state correctness in filtered views

## Out Of Scope

- A separate `Read-it-later` destination
- Lightweight bookmark-only or URL-stub records
- A generalized collections framework
- Mixed local/server media browsing
- Sync, dual-write, mirror, or reconciliation semantics
- Reusing server reminder/feed state as local notification ownership
- A broad Media or Ingest UI redesign
- A generic server jobs console
- Server ingestion-source delete
- Saved searches, digest schedules, reading note links, or advanced server reading workflows
- Study packs, watchlists, research sessions, or writing-suite implementation work

## Approaches Considered

### Option A: `Read-it-later` as a separate destination

This would create a new top-level screen or destination for saved reading.

Why not chosen:

- duplicates media identity and detail behavior
- creates pressure for a second storage model
- makes source-scoped local/server behavior harder to reason about
- conflicts with the desire to keep `Read-it-later` as a focused filtered view over real records

### Option B: Generic collections-first abstraction

This would introduce a new collection/membership abstraction now and represent `Read-it-later` through that generalized system.

Why not chosen:

- too abstract for the first slice
- overfits future collection ideas before sync and membership semantics exist
- conflicts with the fixed decision that `Read-it-later` is a saved-state/filter over real records

### Option C: Seam-first combined vertical

This extends the current media/reading seam so `Read-it-later` and server ingestion-source management both ride the same source-aware contracts.

Why chosen:

- matches the runtime-policy and source-authority work that already landed
- reuses the normalized media seam that already exists
- keeps one record identity model
- avoids inventing a second browsing surface or collection system
- provides a clean base for later watchlists and research work

## Chosen Model

This vertical keeps one source-scoped media product with two user-facing browse subviews:

- `All`
- `Read-it-later`

Those subviews are not separate storage systems. They are alternate filtered views over one normalized media/reading contract.

The core rule is:

- in `local` mode, the user browses and mutates local records plus local `Read-it-later` saved-state
- in `server` mode, the user browses and mutates server reading/item state through the server compatibility mapping

The `Media` screen owns browsing and reading interactions. The `Ingest` screen owns acquisition and source-management interactions. Both consume the same runtime-policy-guided source selection and the same media/reading scope service seam.

## Architecture

### 1. Authority Model

This vertical must inherit the runtime-policy authority model rather than inventing a second media-specific notion of backend state.

The controlling rules are:

- runtime policy decides whether the active source is `local` or `server`
- UI code consumes one source-aware media scope service
- local mode is locally authoritative
- server mode is server-authoritative
- no successful UI mutation is committed until the active authority confirms success

### 2. Core Units

The vertical should remain decomposed into small, explicit units:

- `normalized media contract`
  - outward-facing record shape
  - canonical string-first IDs
  - capability flags
  - saved-state compatibility fields
- `local media adapter`
  - maps local DB rows and local user-state into the normalized contract
  - owns local `Read-it-later` persistence
- `server media adapter`
  - maps server reading/item responses into the normalized contract
  - owns server compatibility mapping for `Read-it-later`
- `media/reading scope service`
  - routes all browse/detail/mutate behavior by active source
  - enforces source-aware authority and unsupported-operation behavior
  - exposes intent-level `Read-it-later` actions rather than forcing UI code to compose low-level status mutations directly
- `media runtime state`
  - owns source-scoped caches, selection, and subview state
- `media screen boundary`
  - translates existing widget behavior into normalized record operations
- `media ingest screen boundary`
  - exposes local ingest in local mode
  - exposes current server ingestion-source management in server mode

### 3. Canonical Identity

The canonical outward-facing record ID remains:

- `<backend>:<entity_kind>:<source_id>`

Examples:

- `local:media:42`
- `server:reading_item:118`
- `server:ingestion_source:7`

The normalized contract must continue to preserve:

- `backend`
- `entity_kind`
- `source_id`
- `backing_media_id`

This is required so the UI never conflates:

- local media records
- server reading items
- ingestion sources
- future file artifacts

### 4. Normalized Record Contract

The normalized media/reading record should expose:

- `id`
- `backend`
- `entity_kind`
- `source_id`
- `backing_media_id`
- `uuid`
- `title`
- `media_type`
- `author`
- `url`
- `created_at`
- `updated_at`
- `status`
- `deleted`
- `is_trash`
- `has_transcript`
- `has_chunks`
- `reading_progress`
- `supports_read_it_later`
- `is_read_it_later`
- `read_it_later_saved_at`

Contract notes:

- `supports_read_it_later` is explicit; it is not inferred from route family or widget location.
- `media` and `reading_item` are the primary saved-for-later-capable kinds in this slice.
- `ingestion_source` is not a saved-for-later-capable kind.
- `is_read_it_later` is a compatibility field:
  - local mode reflects local saved-state
  - server mode reflects compatibility mapping from server reading/item state
- `read_it_later_saved_at` is optional:
  - local mode may populate it from local saved-state persistence
  - server mode should leave it unset unless the server later exposes a real saved-timestamp field

### 5. Server Compatibility Mapping

The server does not expose a native `read_it_later` boolean. Therefore the design must define one explicit compatibility mapping:

- server `save to Read-it-later` => update server item to `status="saved"`
- server `remove from Read-it-later` => update server item to `status="archived"`

This slice must not:

- treat `favorite` as `Read-it-later`
- invent a separate local server-shadow field
- leave the reverse action ambiguous

`Read-it-later` in server mode is therefore a presentation-layer compatibility view over the server’s reading lifecycle state.

This slice must also not:

- synthesize `read_it_later_saved_at` from server `updated_at`
- imply the server owns a precise saved timestamp when it does not currently expose one

### 6. Local Persistence

Local `Read-it-later` should use a narrow user-state persistence model separate from the main media row and separate from local reading progress.

Recommended local storage shape:

- one local-only table keyed by local media ID
- fields:
  - `media_id`
  - `is_read_it_later`
  - `saved_at`
  - `updated_at`

This is preferred over:

- a boolean bolted directly onto the core media row
- reusing reading-progress storage
- inventing a full collection-membership system

### 7. Local Lifecycle Rules

Local saved-state must follow explicit lifecycle rules:

- hard delete cascades removal of local saved-state
- trashed/deleted records do not appear in `Read-it-later` by default
- later trash-specific views, if they exist, remain separate from the `Read-it-later` subview

This prevents the filtered view from becoming a hidden trash surface.

### 8. Source-Native Filtering

The `Read-it-later` subview is a user-facing filter, but the underlying fetch behavior must be backend-native.

Local mode:

- the browse path must query through local persistence-aware filters
- it must not fetch all rows and then only hide non-saved ones in memory

Server mode:

- the browse path must query the server using compatible request shaping, specifically `status=saved`
- it must not fetch generic reading items and then locally filter them into a saved view
- because current server media-type narrowing still relies on client-side shaping in the chatbook stack, first-slice server `Read-it-later` should be exposed under aggregate `All Media`, not as a guaranteed-correct per-type saved view

This is required so:

- paging is correct
- counts are correct
- result sets are stable
- server browsing remains efficient and honest

## UI Design

### 1. Media Screen

`Media` remains the primary browsing destination.

It gains one source-scoped subview/filter:

- `All`
- `Read-it-later`

Subview behavior:

- switching subviews preserves the active source
- switching subviews uses source-native fetches
- if the current selection no longer belongs in the filtered result set, selection is cleared or advanced predictably

The detail surface remains structurally familiar. It simply consumes the normalized saved-state fields and source-aware actions.

### 2. Record-Level Actions

Saved-state behavior must remain source-aware:

- local mode:
  - use explicit scope-service actions such as `save_to_read_it_later(...)` and `remove_from_read_it_later(...)`
  - persist local saved-state
  - update UI only after local persistence succeeds
- server mode:
  - use the same explicit scope-service actions, routed to server authority
  - call server update path using the compatibility mapping
  - update UI only after server success

Browse paths for the saved subview should likewise use an explicit scope-service operation such as `list_read_it_later(...)` rather than ad hoc widget-level filter composition.

If the normalized record does not support `Read-it-later`:

- hide or disable the action
- show explicit unsupported messaging if invoked indirectly

### 3. Reading Progress

Reading progress remains subordinate state on normalized records and continues to use the existing `backing_media_id` rules.

The important rule is:

- `Read-it-later` does not create a second progress model
- saved-for-later records use the same progress seam as any other supported media/reading record

### 4. Ingest Screen

`media_ingest_screen` keeps its role, but its behavior becomes explicitly source-shaped.

In local mode:

- existing local ingest remains authoritative and actionable
- server ingestion-source controls are explanatory or disabled

In server mode:

- the screen exposes the full currently-supported server ingestion-source management contract:
  - create source
  - list sources
  - inspect one source
  - inspect source items
  - patch mutable settings
  - trigger sync
  - upload archive for archive-backed sources where supported
- first-slice create affordances are limited to:
  - `archive_snapshot`
  - `git_repository`
- `local_directory` is deferred or placed behind advanced/server-local handling rather than presented as a normal remote-client create path

This slice explicitly does not promise:

- source delete
- a general server jobs console
- broad server ingest-job management unrelated to sources

### 5. Runtime State

The existing media runtime-state seam should explicitly own:

- active source/backend
- active browse subview
- selected normalized record ID
- current search/filter inputs
- browse results cache
- detail cache
- reading-progress cache
- ingestion-source detail cache
- ingestion-source items cache

On backend switch:

- clear selection
- clear browse/detail/progress/source caches
- preserve only safe UI defaults
- refetch from the newly active source

On subview switch:

- preserve active source
- preserve compatible search/filter inputs where valid
- clear selection when the filtered result set no longer contains the selected record

## Error Handling

This vertical should fail explicitly and source-correctly.

Important failure classes:

- wrong source
- unsupported record capability
- missing progress target
- server unavailable
- server mutation rejected
- stale selection after filter/source change

Required error-handling rules:

- never silently fall back from a failed server write to a local write
- never leave optimistic UI state behind after failed server mutation
- never imply cached authority for remote-only or unavailable server operations
- use concise source-explicit notifications

Representative user messages:

- `Saved to Read-it-later (Local).`
- `Saved to Read-it-later (Server).`
- `Removed from Read-it-later (Server).`
- `Server ingestion sources require server mode.`
- `This record cannot be saved for later.`
- `Reading progress is not available for this record.`
- `Server media backend is unavailable.`

## Testing Strategy

This vertical needs coverage in five layers.

### 1. Normalizer And Contract Tests

- normalized records include saved-state fields
- capability flags are correct for entity kinds
- local and server records present one stable outward contract
- server compatibility mapping normalizes correctly
- `backing_media_id` behavior remains correct

### 2. Local Persistence Tests

- save/remove local `Read-it-later`
- source-native local browse filtering
- hard-delete cascade behavior
- hidden-by-default trash/delete behavior
- local reading-progress non-regression

### 3. Server Adapter And Scope-Service Tests

- server `status=saved` list/search behavior
- save/remove compatibility mapping:
  - save => `saved`
  - remove => `archived`
- existing server-backed workspace or source scope metadata remains preserved when save/remove paths resave records
- server `read_it_later_saved_at` remains unset unless a real server field exists
- unsupported-operation behavior remains explicit
- ingestion-source create support lands through the same seam
- server `Read-it-later` subview is only exposed where aggregate `All Media` semantics are correct
- existing list/detail/update/items/sync/archive flows remain covered

### 4. UI State Tests

- `Media` subview switch updates results correctly
- removing an item from the saved filter updates filtered state correctly
- backend switch clears stale selection and caches
- `media_ingest_screen` presents local vs server behavior correctly
- failed server mutation does not leave false saved-state in the UI

### 5. Representative Integration Slices

- local browse -> save -> filter -> remove
- server browse -> save -> filter -> remove
- local/server backend switch with active selection and filter
- server ingestion-source create/list/detail/update/items/sync/archive upload

## Rollout Order

The implementation should land internally in this order:

1. extend normalized record and adapter contracts
2. add local saved-state persistence
3. add server compatibility mapping for save/remove and source-native list filtering
4. extend scope-service and server media service for create plus saved-state actions
5. wire `Media` subview/filter behavior
6. wire `media_ingest_screen` to the full current server contract
7. add regression coverage
8. update parity docs after verification

## Success Criteria

This vertical is complete when:

- `Read-it-later` is a real source-scoped subview inside `Media`
- local mode supports save/remove/filter through local authority
- server mode supports save/remove/filter through the explicit `saved`/`archived` compatibility mapping
- server mode does not invent a fake saved timestamp and does not overpromise per-media-type saved filtering
- source-native filtering is used in both local and server mode
- `media_ingest_screen` exposes the full currently-supported server ingestion-source management contract
- first-slice server source creation is limited to `archive_snapshot` and `git_repository`
- no covered path bypasses the scope service
- runtime-policy and source-authority rules remain intact
- parity docs can honestly move this row from a major gap to a substantial landed seam
