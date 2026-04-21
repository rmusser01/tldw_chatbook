# tldw_chatbook RAG, Chunking, And Template Alignment Vertical Design

**Date:** 2026-04-20  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next interoperability and parity vertical after media/files/ingestion: align `tldw_chatbook` chunking-template management, embedding-collection management, and the first retrieval-admin compatibility seams with the current `tldw_server` contracts while preserving `tldw_chatbook` as a standalone local-first application.

## Context

`tldw_chatbook` already has substantial local retrieval infrastructure. This is not a blank area of the product. The local side includes:

- `tldw_chatbook/Chunking/chunking_templates.py`
- `tldw_chatbook/Chunking/chunking_interop_library.py`
- `tldw_chatbook/RAG_Search/chunking_service.py`
- `tldw_chatbook/RAG_Search/late_chunking_service.py`
- `tldw_chatbook/Embeddings/Embeddings_Lib.py`
- `tldw_chatbook/Embeddings/Chroma_Lib.py`
- `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
- `tldw_chatbook/Widgets/chunking_templates_widget.py`
- `tldw_chatbook/UI/Embeddings_Management_Window.py`

The local side already supports:

- local chunking template files and DB-backed template records
- local per-document chunking configuration in the media DB
- local Chroma-backed embedding collections
- local retrieval configuration and RAG execution pipelines
- a Textual chunking-template manager and a Textual embeddings manager

At the same time, the current server-side surface has evolved and now exposes richer and more explicit contracts:

- `/api/v1/chunking/templates/*`
- `/api/v1/chunking/capabilities`
- `/api/v1/embeddings/collections`
- `/api/v1/embeddings/collections/{collection_name}`
- `/api/v1/embeddings/collections/{collection_name}/stats`
- richer RAG request shaping in the newer UI service layer and server workflow adapters

The immediate parity problem is not that `tldw_chatbook` lacks retrieval code. The immediate parity problem is that the current local admin surfaces do not have a mode-aware local/server compatibility seam:

- `tldw_api` does not yet expose chunking-template or embedding-collection contracts
- `ChunkingTemplatesWidget` only talks to the local media DB service
- `EmbeddingsManagementWindow` has a working model pane, but its collection pane is still largely placeholder logic
- `SearchRAGWindow` mixes older local profile concepts with partially stale search-call signatures and is too broad to use as the first compatibility surface

This vertical therefore focuses on the admin/control seam first, not on rewriting the search execution surface.

## Product Decisions

The following decisions are fixed for this vertical:

- This vertical covers all of the following together:
  - chunking-template compatibility
  - embedding-collection compatibility
  - retrieval-admin compatibility seams needed by those surfaces
- `tldw_chatbook` remains a standalone local-first application.
- `Server mode` is the reference behavior where server support exists.
- In `server mode`, reads and writes that the server supports are server-authoritative.
- In `server mode`, the client updates local UI state only after confirmed server success.
- In `local mode`, existing local chunking-template and embedding-collection behavior remains authoritative.
- No sync, dual-write, or cross-scope copy behavior is introduced in this vertical.
- `ChunkingTemplatesWidget` is the primary template-management surface for this vertical.
- `EmbeddingsManagementWindow` is the primary embedding-collection-management surface for this vertical.
- `SearchRAGWindow` stays visually and behaviorally unchanged in this slice except for any non-invasive compatibility hooks strictly required by shared services.
- Local chunking templates and server chunking templates are treated as separate backends with one normalized UI contract, not as one merged storage namespace.
- Local embedding collections and server embedding collections are treated as separate backends with one normalized UI contract, not as one merged storage namespace.
- Template identity is normalized around template `name` as the cross-backend outward-facing key for this slice.
- Embedding collection identity is normalized around collection `name` as the outward-facing key for this slice.
- Local integer template IDs remain local implementation details and must not become the main UI contract across backends.
- Server-only chunking-template features that do not exist locally, such as diagnostics or apply/match/learn flows, may be exposed through the service seam later but are not required to be first-class in the TUI in this slice.
- Server-side RAG request parity is deferred to a later search-execution vertical.
- Search/chat retrieval behavior is explicitly out of scope unless a minimal shared contract is needed to avoid blocking later work.

## In Scope

- Add `tldw_api` schemas and client coverage for:
  - chunking template list/get/create/update/delete
  - chunking template apply and diagnostics where helpful for parity completeness
  - embedding collection list/delete/stats
- Add a mode-aware retrieval-admin service seam parallel to the notes/media/character service seams.
- Add normalization helpers that converge local template rows and server template payloads onto one template-facing contract.
- Add normalization helpers that converge local collection metadata and server collection payloads onto one collection-facing contract.
- Refactor `ChunkingTemplatesWidget` so it can operate against local or server backends without changing its basic interaction model.
- Refactor `EmbeddingsManagementWindow` so the collection pane actually loads real collections, shows real detail/stats, and can delete collections in local or server mode.
- Add focused tests for:
  - `tldw_api` schema/client coverage
  - normalization behavior
  - local/server service routing
  - chunking template widget backend-aware behavior
  - embeddings collection-management behavior
- Update parity docs after the vertical lands.

## Out Of Scope

- Rewriting `SearchRAGWindow` search execution
- Replacing local RAG pipelines with server-backed execution
- Sync or dual-write semantics
- Cross-backend copy/move/import flows for templates or collections
- Reworking local chunking internals beyond the seam changes needed to support normalization
- Reworking local embedding model download and load/unload behavior
- New visual redesign of the retrieval/search product
- Rich chunking playground parity
- Broad RAG settings parity
- Workspace-scoped retrieval UX
- Study/evals/recipe parity

## Approaches Considered

### Option A: Search-first RAG parity

This would start by rewriting `SearchRAGWindow` and aligning server-side RAG query payloads first.

Why not chosen:

- The current search window is broader and noisier than the admin surfaces
- It mixes old local concepts and partially stale call shapes
- It would turn this branch into a general search rewrite instead of a bounded parity slice

### Option B: Templates-only parity

This would align chunking templates first and leave embedding collections for later.

Why not chosen:

- Leaves one of the most visible local/server admin gaps open
- Forces a second service seam pass later for collections
- Misses the opportunity to land one coherent retrieval-admin contract now

### Option C: Compat-first retrieval-admin vertical

This treats server chunking-template and embedding-collection contracts as the reference, normalizes local and server data onto one shared admin contract, and updates the two existing TUI management surfaces without changing search execution.

Why chosen:

- Keeps the vertical bounded and testable
- Lands concrete user-visible parity in existing screens
- Avoids destabilizing the broader search UI
- Produces reusable local/server seams that a later search-execution vertical can build on

## Chosen Model

This vertical treats the server retrieval-admin contracts as the compatibility reference while preserving local chunking templates and local Chroma collections as first-class standalone implementations.

That means:

- local templates remain real local records
- server templates remain real server records
- local collections remain real local collections
- server collections remain real server collections
- the UI does not pretend those backends are one shared storage layer
- the UI does consume one normalized admin contract across both
- mode determines the active backend
- later sync or import/export work can build on explicit normalized identities instead of widget-local assumptions

## Architecture

### 1. Backend Mode Ownership

This vertical uses the existing application runtime-backend selection model instead of introducing a retrieval-specific toggle.

The retrieval-admin surfaces must:

- consume the current runtime backend when available
- default to `local` if no backend state is available
- refresh their list/detail state on backend changes
- avoid caching local and server records in one mixed list

This is intentionally the same local-first pattern already used in the notes, characters, and media verticals.

### 2. Client And Schema Layer

This vertical requires extending `tldw_chatbook/tldw_chatbook/tldw_api/` beyond media-processing routes.

The design requires:

- schema models for server chunking-template payloads
- schema models for server embedding-collection payloads
- async client methods for:
  - list/get/create/update/delete chunking templates
  - apply chunking template
  - chunking-template diagnostics
  - list embedding collections
  - delete embedding collection
  - get embedding collection stats

The client layer remains a thin contract wrapper. It should not contain normalization policy, TUI-specific logic, or local fallback behavior.

### 3. Retrieval-Admin Service Seam

This vertical requires a dedicated mode-aware service seam so UI code does not know whether it is talking to:

- the local media DB-backed template service
- local Chroma collections
- server chunking-template routes
- server embedding-collection routes

The design requires:

- a local adapter for chunking-template operations built on `ChunkingInteropService`
- a local adapter for embedding-collection operations built on `ChromaDBManager`
- a server adapter built on the new `tldw_api` methods
- one scope service that routes by backend and returns normalized records

The seam should normalize template records to a stable shape like:

```python
{
    "backend": "local" | "server",
    "record_type": "chunking_template",
    "record_id": "<backend>:chunking_template:<name>",
    "name": "...",
    "description": "...",
    "template_name": "...",
    "tags": [...],
    "is_builtin": bool,
    "version": int | None,
    "created_at": "...",
    "updated_at": "...",
    "raw_payload": {...},
}
```

and collection records to a stable shape like:

```python
{
    "backend": "local" | "server",
    "record_type": "embedding_collection",
    "record_id": "<backend>:embedding_collection:<name>",
    "name": "...",
    "document_count": int | None,
    "embedding_dimension": int | None,
    "status": "ready" | "unknown",
    "metadata": {...},
    "raw_payload": {...},
}
```

### 4. Template Identity And Data Mapping

Local templates currently expose:

- integer `id`
- `name`
- `description`
- `template_json`
- `is_system`
- timestamps

Server templates currently expose:

- integer `id`
- `uuid`
- `name`
- `description`
- `template_json`
- `is_builtin`
- `tags`
- timestamps
- `version`
- `user_id`

The compatibility seam must:

- normalize `is_system` to `is_builtin`
- normalize absent local `tags` to `[]`
- normalize absent local `version` to `None` or a fixed local value where needed
- keep local integer IDs as adapter details, not cross-backend UI identity
- treat `name` as the compatibility key for CRUD and selection

### 5. Collection Identity And Data Mapping

Local collections are currently represented through `ChromaDBManager` and collection metadata, while the current TUI collection pane still uses placeholder records.

Server collections expose:

- `name`
- `metadata`
- separate stats endpoint for `count` and `embedding_dimension`

The compatibility seam must:

- normalize local collection metadata into the same record shape
- optionally enrich server collection rows with stats when the stats endpoint is requested
- avoid assuming all stats are available from the list endpoint
- keep collection `name` as the outward-facing key

### 6. UI Integration

#### `ChunkingTemplatesWidget`

This widget keeps its current role and general layout.

The refactor must:

- resolve a scope service from the app instance instead of talking directly to `ChunkingInteropService`
- refresh records for the active backend
- use normalized template records for selection and detail rendering
- route create/edit/delete through the active backend
- disable or explicitly message unsupported operations if needed

The widget should not be redesigned in this slice.

#### `EmbeddingsManagementWindow`

This window keeps its current model-management pane, but its collection pane is no longer allowed to remain placeholder logic.

The refactor must:

- actually load collections in local mode
- actually load collections in server mode
- show collection detail and stats using normalized records
- support single and batch collection deletion where the backend supports it
- continue to keep model download/load/unload behavior local-only

If a capability is local-only, the window must state that explicitly instead of failing silently.

### 7. Error Handling

The service seam must distinguish:

- unsupported in current backend
- not found
- validation failure
- transport/server failure

The TUI should translate these into concise user-facing notifications, not stack traces.

Expected examples:

- trying to edit a built-in template in either backend yields a clear warning
- trying to delete a missing server collection yields a not-found warning
- server transport failure yields an error notification and preserves current UI selection state where possible

### 8. Testing Strategy

This vertical uses the same contract-first testing pattern as prior parity slices:

1. `tldw_api` schemas and client methods
2. service normalization and routing
3. widget/window behavior on top of the seam

Targeted tests should verify:

- request parameter shaping
- template and collection normalization
- backend-aware selection behavior
- widget state refresh when backend changes
- collection deletion flows
- unsupported-operation handling

## Risks And Mitigations

### Risk 1: Search scope creep

Because the repo has a large amount of retrieval code, this vertical could easily turn into a general RAG rewrite.

Mitigation:

- keep `SearchRAGWindow` out of the main implementation path
- treat only template and collection management as user-facing scope in this slice

### Risk 2: Local/server template-shape drift

Local templates and server templates do not have the same fields.

Mitigation:

- normalize into one explicit record shape in the service seam
- keep raw payloads available for backend-specific details

### Risk 3: Local collection pane is underimplemented

The current TUI collection pane is placeholder logic and may hide missing local assumptions.

Mitigation:

- treat that as part of the vertical, not as unrelated cleanup
- land real local collection loading and deletion support through the same seam used for server mode

### Risk 4: Runtime backend ownership is implicit

Some retrieval surfaces may not yet read backend mode the same way other parity verticals do.

Mitigation:

- follow the same app-instance runtime-backend access pattern used in the media vertical
- keep backend choice outside the widget itself

## Deliverables

- `tldw_api` support for chunking-template and embedding-collection contracts
- a new mode-aware retrieval-admin service seam
- backend-aware `ChunkingTemplatesWidget`
- backend-aware `EmbeddingsManagementWindow` collection pane
- focused automated coverage for client/schema/service/UI behavior
- updated parity docs once the vertical lands

## Follow-On Work Explicitly Deferred

- search execution parity in `SearchRAGWindow`
- richer chunking playground parity
- broader RAG settings parity
- sync/dual-write for templates and collections
- workspace-aware retrieval UX
- study/evals/retrieval recipe parity
