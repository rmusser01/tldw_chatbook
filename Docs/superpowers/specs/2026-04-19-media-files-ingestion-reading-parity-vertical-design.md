# tldw_chatbook Media, Files, Ingestion, And Reading Parity Vertical Design

**Date:** 2026-04-19  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next interoperability and parity vertical after characters/personas: align `tldw_chatbook` media browsing, ingestion-source management, reading-item compatibility, and reading-progress behavior with the current `tldw_server` contracts while preserving `tldw_chatbook` as a standalone local-first application.

## Context

`tldw_chatbook` already has a substantial local media system. It is not a placeholder and it is not a thin cache of server data. The local side includes:

- `tldw_chatbook/DB/Client_Media_DB_v2.py`
- `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
- `tldw_chatbook/Local_Ingestion/audio_processing.py`
- `tldw_chatbook/Local_Ingestion/video_processing.py`
- `tldw_chatbook/UI/MediaWindow_v2.py`
- `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
- `tldw_chatbook/UI/Screens/media_screen.py`
- `tldw_chatbook/UI/Screens/media_ingest_screen.py`

The local media stack already supports a wide range of capabilities:

- local media item CRUD and search
- integer primary keys plus local UUIDs
- transcripts
- chunking and embedding-adjacent state
- document versions
- keyword relationships
- trash and soft-delete semantics
- local ingestion workflows
- local sync log/version metadata

At the same time, the current server-side content surface has evolved and is now split across several newer API families:

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/files.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py`

Those route families represent at least four related but separate concepts:

- structured file artifacts
- user-managed ingestion sources and source items
- reading items and reading-library workflows
- reading progress keyed to underlying media documents

`tldw_chatbook`’s current `tldw_api` client still mainly exposes the older `/api/v1/media/*` surface. That means there is currently no mode-aware local/server compatibility seam for the newer server content model. The existing local media screens therefore cannot switch cleanly between:

- local media records
- server reading items
- server-backed reading progress
- server ingestion-source management

This is the parity gap this vertical addresses.

## Product Decisions

The following decisions are fixed for this vertical:

- This vertical covers all of the following together:
  - media browsing compatibility
  - file artifact compatibility
  - ingestion-source compatibility
  - reading-item compatibility
  - reading-progress compatibility
- `tldw_chatbook` remains a standalone local-first application.
- `Server mode` is the reference behavior where server support exists.
- In `server mode`, reads and writes that the server supports are server-authoritative.
- In `server mode`, the client updates local UI/cache only after confirmed server success.
- In `local mode`, existing local media behavior remains authoritative by default.
- There is no mixed local/server media list in this vertical. Mode determines the active source.
- `media_screen` remains the primary browsing surface.
- `media_ingest_screen` remains the primary ingestion-management surface.
- The vertical preserves the current media product layout, but it does include targeted internal refactors where needed to introduce a real mode-aware seam.
- Media backend ownership must be explicit for this vertical; it must not rely on ad hoc widget-local assumptions.
- No new media-specific mode toggle is introduced in this slice. Media surfaces consume the existing runtime backend selection when available and default to `local` otherwise.
- The server `files` domain is included for compatibility, but is not promoted to a top-level browsed library in this slice.
- The server `reading` domain is included in the first implementation slice, not deferred.
- The server `reading_progress` domain is included in the first implementation slice, not deferred.
- Reading progress is subordinate state keyed by media identity plus backend, not a top-level browsed entity.
- Canonical outward-facing IDs are string-first.
- Canonical normalized IDs use the shape `<backend>:<entity_kind>:<source_id>`.
- `entity_kind` is explicit in normalized contracts and must not be inferred from field presence.
- Expected entity kinds for this vertical are:
  - `media`
  - `reading_item`
  - `ingestion_source`
  - `file_artifact`
- Local integer IDs remain storage details, not UI contracts.
- The existing local batch-ingestion workflow is not mirrored to the server in this slice.
- In `server mode`, `media_ingest_screen` becomes a minimal ingestion-source management surface inside the current shell.
- The legacy `Remote (TLDW API)` ingest path is retired as a direct-client execution path in this vertical.
- The existing second ingest-panel slot becomes backend-aware:
  - in `server mode`, it hosts scope-service-backed ingestion-source management
  - in `local mode`, it is non-actionable and explains that server ingestion sources require server mode
- Reading saved searches, reading digest schedules, reading note links, reading import jobs, and server TTS/summarization flows are out of scope unless needed incidentally for a shared contract.
- Sync, dual-write, and local/server reconciliation remain out of scope.
- RAG/chunk/template parity remains out of scope beyond metadata hooks needed to avoid blocking later work.

## In Scope

- Add `tldw_api` schemas and client coverage for:
  - file artifacts
  - reference image listing where relevant to file-artifact compatibility
  - ingestion source CRUD/list/detail
  - ingestion source item listing
  - ingestion source sync trigger
  - ingestion source archive upload
  - reading item list/detail/update/delete coverage needed for compatibility
  - reading progress get/update/delete
- Add a server-backed media/reading service layer parallel to the existing notes/chatbooks/character service seams.
- Add a mode-aware media/reading scope service used by media UI code.
- Add normalization helpers that converge local rows and server payloads onto one media-facing contract.
- Add a narrow local reading-progress store so offline/local mode can carry comparable progress state.
- Add explicit media runtime-state wiring so both media screens resolve the same backend and invalidate state consistently.
- Refactor `MediaWindow_v2` at the window boundary so it can consume normalized IDs and scope-service operations without a visual redesign.
- Refactor `MediaIngestWindowRebuilt` at the window boundary so server-mode source management flows through the scope service instead of direct `TLDWAPIClient` usage.
- Update `media_screen` to browse server-mode normalized records and inspect/update reading progress through the scope service.
- Update `media_ingest_screen` to support server-mode ingestion-source inspection and management through the scope service.
- Add regression coverage for:
  - schema and client route coverage
  - normalization behavior
  - local/server routing
  - media runtime-state ownership and invalidation behavior
  - backend-change invalidation in media state
  - server-mode reading progress flows
  - server-mode ingestion-source flows
- Update parity docs once the vertical lands.

## Out Of Scope

- Broad visual or product redesign of `MediaWindow_v2` or `MediaIngestWindowRebuilt`
- Rewriting child media widgets beyond the boundary changes needed to remove direct backend coupling
- Replacing the local media DB with a server cache
- Sync or dual-write semantics
- Background job center work
- Full file-artifact export/download UX
- Full server-side archive-management UX beyond the minimum upload trigger
- Server reading saved searches
- Server reading digest schedules and outputs
- Reading note links
- Reading import/export job UI
- Summary, TTS, or other advanced reading workflows
- Embeddings, chunking-template, or chat-document parity beyond compatibility hooks

## Approaches Considered

### Option A: Keep media local and add only server ingestion helpers

This would keep media browsing local-only and add a small server ingestion-source feature set next to it.

Why not chosen:

- Leaves `reading` and `reading_progress` structurally disconnected
- Forces media identity to be defined again later
- Preserves too much drift between local and server content workflows

### Option B: Full-stack media parity in one pass

This would attempt to align files, ingestion, reading, reading progress, advanced reading workflows, file downloads, and more in one branch.

Why not chosen:

- Too broad for a single parity slice
- High risk of UI and service sprawl
- Weakens testability and reviewability

### Option C: Contract-first media/files/ingestion/reading vertical

This treats the server content model as the reference, normalizes local and server data onto one shared contract, and updates the existing screens with contained boundary refactors instead of a visual redesign.

Why chosen:

- Produces one stable media identity seam instead of several partial ones
- Preserves standalone local behavior
- Matches the current parity pattern already used for chats, notes, chatbooks, and characters/personas
- Lets later sync work build on explicit contracts rather than UI assumptions

## Chosen Model

This vertical treats the newer server content families as the reference model for compatibility while preserving local media as a first-class offline implementation.

That means:

- local media records remain real, local-first entities
- server reading items remain real, server-first entities
- the UI does not pretend they are the same storage object
- the UI does consume one normalized browse/detail contract across both
- reading progress hangs off the normalized record where supported
- ingestion sources remain a separate management domain surfaced through the existing ingest screen
- file artifacts are covered in the adapter layer now so the later UX does not require reworking the identity seam

The goal is not to force the server’s internal decomposition into the local UI. The goal is to expose one coherent boundary where:

- the media screen knows how to browse the active backend
- the ingest screen knows how to manage the active backend
- the service layer knows which operations are supported in each mode
- later sync work inherits explicit IDs and lifecycle boundaries

## Architecture

### 1. Backend Mode Ownership

The media surfaces currently do not have an explicit state owner for backend selection. This vertical must make backend ownership explicit instead of inferring it ad hoc inside individual widgets.

The design requires:

- one small media runtime-state holder, likely alongside the media screen wrappers
- one authoritative `runtime_backend` value shared by `media_screen` and `media_ingest_screen`
- initialization from the existing runtime backend selection when available, with `local` as the fallback default
- explicit invalidation of media-derived UI state when the backend changes

This runtime state owns only media-facing UI state, such as:

- selected normalized record ID
- active media type and search filters
- cached list/detail payloads
- reading-progress cache
- ingestion-source detail cache

On backend change, the media surfaces must clear:

- selected record/detail state
- reading-progress cache
- ingestion-source cache
- stale panel-specific derived state

before refetching from the newly selected backend.

### 2. Service Layer

This vertical requires a dedicated media/reading service seam. UI code should not call `tldw_api` methods directly and should not know about separate server route families.

The design requires:

- `tldw_chatbook/tldw_api/media_reading_schemas.py`
- `tldw_chatbook/Media/server_media_reading_service.py`
- `tldw_chatbook/Media/media_reading_scope_service.py`
- a small normalization module, likely `tldw_chatbook/Media/media_reading_normalizers.py`
- a small media runtime-state module near the screen wrappers

The service structure should mirror the pattern already used elsewhere in the repo:

- thin server adapter
- local/server facade
- UI consuming one mode-aware seam

### 3. Canonical Identity

The normalized seam must expose string-first IDs and explicit kinds.

Normalized ID format:

- `<backend>:<entity_kind>:<source_id>`

Examples:

- `local:media:42`
- `server:reading_item:118`
- `server:ingestion_source:7`
- `server:file_artifact:23`

Normalized records must also carry:

- `backend`
- `entity_kind`
- `source_id`
- `backing_media_id` when the record supports reading-progress operations

This is required so the UI and service layer do not conflate:

- a local media row
- a server reading item
- a server file artifact

just because all three represent content-adjacent objects.

`backing_media_id` is the raw backend-native media identifier used to resolve reading progress.

- For local media rows, it will usually equal the local media row ID.
- For server reading items, it must carry the underlying server `media_id`, because server reading-progress routes are keyed by media ID rather than reading-item ID.
- For entity kinds that do not support reading progress in this slice, it may be `None`.

### 4. Shared Media-Facing Contract

The media-facing normalized record should expose a stable shape regardless of backend:

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

Notes:

- For local rows, `entity_kind` will usually be `media`.
- For server browse rows in this slice, `entity_kind` will usually be `reading_item`.
- `backing_media_id` is the raw progress target, not another canonicalized record ID.
- `uuid` is preserved when present but is not the canonical outward-facing ID.
- `reading_progress` is nested normalized progress data or `None`.

### 5. Shared Reading-Progress Contract

Normalized reading-progress state should expose:

- `backend`
- `backing_media_id`
- `current_page`
- `total_pages`
- `percent_complete`
- `view_mode`
- `zoom_level`
- `cfi`
- `last_read_at`

This contract is backend-neutral even though the underlying store differs. The important constraint is that progress operations always have the raw backend-native media identifier they need.

### 6. Local Reading Progress

Current local media behavior does not appear to expose a comparable reading-progress contract. That is a parity hole and should not remain server-only.

This vertical should add a narrow local reading-progress store keyed by local media ID. The design should keep it focused:

- one table or equivalent storage layer in the local media DB
- one fetch helper
- one upsert helper
- one delete helper
- local-only persistence semantics with no sync participation in this slice

Local reading-progress rows are explicitly:

- excluded from `sync_log`
- excluded from entity versioning and document-version history
- excluded from sync, dual-write, and conflict-resolution semantics in this slice

This is not a broad reading-feature redesign. It is a minimal offline parity primitive so the mode-aware seam is not asymmetrical by design.

### 7. UI Boundary Adaptation

The current media windows are directly coupled to integer IDs, `app_instance.media_db`, and legacy direct API usage. This vertical must budget a targeted boundary refactor. That refactor is in scope even though the visible layout remains recognizable.

Required adaptation rules:

- `MediaWindow_v2` remains the orchestrator, but it becomes the translation boundary between existing child-widget events and the normalized media contract.
- The selected record tracked by the window becomes a normalized string-first ID rather than a raw integer local media ID.
- Existing child widgets may continue to emit local integer IDs where changing that contract would create excessive churn; the window boundary translates them into normalized IDs and scope-service calls.
- Search, detail load, delete/undelete, metadata update, analysis save, document-version save/delete, and reading-progress operations should stop reaching directly into `app_instance.media_db` from the window once the seam is in place.
- `MediaIngestWindowRebuilt` keeps the current shell, but the server-facing panel stops constructing `TLDWAPIClient` directly and routes all server operations through the scope service.

This is a boundary and state-seam refactor, not a mandate to rewrite every child widget in the media stack.

### 8. Screen Responsibilities

#### `media_screen`

`media_screen` remains the primary browsing surface.

In `local` mode:

- existing local media browsing behavior remains authoritative
- the screen reads through the new scope service instead of reaching directly into local-only assumptions
- the screen tracks selection by normalized record ID even when the underlying local row ID is still integer-native

In `server` mode:

- the screen lists normalized server-backed browse rows
- detail inspection comes through the same scope service
- reading progress get/update flows through the same scope service
- backend changes invalidate current selection and cached results through the shared media runtime state
- no mixed local/server list is allowed

#### `media_ingest_screen`

`media_ingest_screen` remains the primary ingestion-management surface.

In `local` mode:

- existing local ingestion behavior remains intact
- the server/source panel is present only as explanatory or disabled UI, not as an alternate direct-write path

In `server` mode:

- the screen becomes a minimal ingestion-source management surface inside the existing shell
- the former remote panel slot is repurposed to this server-mode source-management view
- required operations:
  - list sources
  - inspect one source
  - inspect source items
  - trigger sync
  - patch mutable settings
  - upload archive for archive-backed sources where supported

What it explicitly does not do in this slice:

- replicate the full local batch-ingestion wizard against the server
- become a general server jobs console

### 9. Files Domain Boundary

The `files` domain is included now for compatibility but is not elevated to a first-class browsed library in this vertical.

Reason:

- the server has real file-artifact and reference-image workflows
- later UI work will need those contracts
- if the identity seam ignores files now, later file UX will force a contract rewrite

Therefore this slice should include:

- schema coverage
- client coverage
- server service coverage
- normalized identity rules

But only minimal UI exposure if an existing media flow truly needs it.

### 10. Reading Domain Boundary

The server `reading` surface is broader than what the existing media UI needs.

This vertical should include only the subset required for stable compatibility:

- list reading items
- fetch reading item detail
- update reading item fields needed for current compatibility
- delete reading item only if required by current UI parity
- reading progress get/update/delete

This vertical should explicitly defer:

- reading saved searches
- note links
- import jobs
- digest schedules
- TTS
- summarization

Status addendum, 2026-04-23: later parity slices have now landed typed server-backed support for reading saved searches, note links, import jobs, archive creation, export bytes, summarization responses, and TTS bytes at the client/service/scope layer. Digest schedules remain deferred while workflows are out of scope, and mounted UX for the new action endpoints is intentionally left to the parallel UX pass.

## Behavior Matrix

### Local Mode

- media browse: supported
- media detail: supported
- reading progress get/update/delete: supported after the new local parity store lands
- ingestion-source management: unsupported beyond explanatory or disabled UI
- file-artifact management: unsupported
- unsupported operations fail explicitly through the scope service

### Server Mode

- media/reading browse: supported through normalized server rows
- detail inspection: supported
- reading progress get/update/delete: supported where the record exposes `backing_media_id`
- ingestion-source management: supported for the chosen minimal surface
- file-artifact coverage: supported in client/service seam, minimally surfaced in UI

## Error Handling

Unsupported operations should use the same explicit pattern already established in other mode-aware seams:

- local-only unsupported server operations fail with clear local messages
- server-only unsupported local operations fail with clear local messages
- backend unavailability is explicit and non-silent
- backend change invalidates stale selection and cache state before refetch

Examples:

- `Local ingestion sources are not available yet.`
- `Server ingestion sources require server mode.`
- `Server media backend is unavailable.`
- `Select a media item first.`
- `Reading progress is not available for this record.`

The UI should surface these through the existing notification pattern instead of silently ignoring the action.

## Testing Strategy

This vertical needs coverage at four layers.

### 1. `tldw_api` Coverage

- schema validation for file, ingestion-source, reading-item, and reading-progress payloads
- client route wiring for the new endpoints
- request serialization for patch/update payloads

### 2. Normalization Coverage

- local media row to normalized contract
- server reading item to normalized contract
- server file artifact to normalized contract
- local and server reading-progress normalization
- canonical ID generation
- `backing_media_id` mapping for records that support progress
- timestamp normalization

### 3. Scope-Service Coverage

- local/server routing
- explicit unsupported-operation failures
- local reading-progress routing
- local reading-progress non-sync semantics
- server reading-progress routing
- ingestion-source server routing

### 4. UI/Screen Coverage

- media runtime state resolves one backend across both media surfaces
- media screen invalidates selected item and caches on backend change
- server-mode browse/detail path consumes normalized rows
- server-mode reading-progress fetch/update works through the seam
- ingest-screen server mode supports source list/detail/item list/trigger sync/patch/archive upload without broad redesign
- local-mode ingest screen does not execute legacy direct server writes through the old remote path

## Acceptance Criteria

This vertical is complete when all of the following are true:

- `tldw_api` covers the missing `files`, `ingestion-sources`, `reading`, and `reading-progress` endpoints needed for this slice
- the app exposes a mode-aware media/reading service seam
- media backend ownership is explicit and shared across both media screens
- normalized IDs are explicit and string-first
- normalized records expose raw `backing_media_id` where progress is supported
- local media rows and server browse rows can both be rendered through one stable contract
- local mode has a minimal reading-progress store
- server-mode media browsing works without changing the overall media product layout
- server-mode ingest screen can manage ingestion sources through the current shell
- unsupported operations fail explicitly
- parity docs are updated to record the canonical contract and deferred work

## Implementation Sequence

The implementation should follow this order:

1. Add schemas and client coverage for the new server route families.
2. Add normalization helpers, canonical ID rules, and `backing_media_id` mapping.
3. Add local reading-progress storage with explicit local-only, non-sync semantics.
4. Add server service and mode-aware scope service.
5. Add explicit media runtime-state wiring and backend invalidation behavior.
6. Refactor `MediaWindow_v2` at the window boundary to consume the new seam.
7. Refactor `MediaIngestWindowRebuilt` at the window boundary and repurpose the legacy remote slot for server-mode source management.
8. Add docs and regression coverage.

This keeps the vertical contract-first and avoids UI work before the backend boundary is stable.
