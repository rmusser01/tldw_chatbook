# Media And Reading Domain Edge Contract

Date: 2026-04-29

Status: Lane D contract

Related code:

- `tldw_chatbook/Media/media_reading_scope_service.py`
- `tldw_chatbook/Media/media_reading_normalizers.py`
- `tldw_chatbook/Media/server_media_reading_service.py`
- `tldw_chatbook/Media/local_media_reading_service.py`
- `tests/Media/test_media_reading_scope_service.py`
- `tests/Media/test_server_media_ingest_jobs_service.py`
- `tests/Media/test_server_media_reading_service.py`

## Scope

This contract covers non-UI media, ingestion, and reading parity edges.

In scope:

- Ingest-job event/status normalization.
- Read-it-later server aggregate-only behavior.
- Chunk-level TTS adoption decision.
- Saved-view unsupported cases.
- Source-separated local/server media and reading state.
- Required focused service tests.

Out of scope:

- UI/UX redesign.
- Broad UI tests.
- Write sync, dual-write, or mirror replay.
- Replacing local media storage with a server cache.
- Workflows.

## Source Authority

| Edge | Source owner | Contract |
|---|---|---|
| Local media records | Local | Local DB-backed media records remain local-authoritative in local mode. |
| Server reading items | Server | Server reading APIs are authoritative in server mode. |
| Server ingest jobs | Server | Server job submission/status/list/cancel/events are active-server-owned. |
| Local ingest jobs | Local | Local ingest jobs are local-authoritative and must not be silently sent to server mode. |
| Reading progress | Active source | Progress is keyed to the backing media identity of the active source. |
| File artifacts/reference images | Active source | Normalized by backend and entity kind; not merged across sources. |

## Ingest-Job Status And Event Normalization

Ingest jobs must normalize status and events without hiding source identity.

Required job record fields for any future shared model:

- `id`: canonical ID in the shape `<backend>:media_ingest_job:<source_id>` when exposed through shared UI/service seams.
- `backend`: `local` or `server`.
- `entity_kind`: `media_ingest_job`.
- `source_id`: native job ID from the owning backend.
- `batch_id`: native batch/session grouping where available.
- `state`: normalized lifecycle value.
- `status`: raw backend status when available.
- `progress`: structured backend progress object.
- `error`: backend error text/object, never swallowed.
- `created_at` and `updated_at`: timestamps when supplied.

Normalized lifecycle values:

| Normalized state | Accepted source statuses/events |
|---|---|
| `queued` | `queued`, `pending`, `created`, `accepted` |
| `running` | `running`, `processing`, `started`, `in_progress` |
| `completed` | `completed`, `complete`, `succeeded`, `success`, terminal success event |
| `failed` | `failed`, `error`, `errored`, terminal failure event |
| `cancelled` | `cancelled`, `canceled`, terminal cancellation event |
| `unknown` | Missing or unrecognized status |

Event normalization rules:

- Server event observation uses `media.ingestion_jobs.observe.server`.
- Event cursor state belongs to the active server profile and stream/batch; it must not be reused across server profiles.
- Events must include source, stream/batch ID, event ID or fallback dedupe key, received timestamp, payload kind, and normalized entity reference once Lane A shared events are available.
- Unsupported or stale cursors must reset/requery explicitly rather than replaying across streams.
- Terminal status events may update local presentation state but do not create local media records until confirmed by the owning source.

Current service behavior already routes server ingest job submit/detail/list/cancel/batch-cancel/reprocess and event observation through server scope actions.

## Read-It-Later Server Aggregate-Only Behavior

Decision: server read-it-later saved browsing is aggregate-only.

Server mode may expose the `All Media` saved/read-it-later context. Server mode must not pretend per-media-type saved views exist until the server contract supports them.

Rules:

- Local mode supports read-it-later browsing without aggregate-only restrictions.
- Server mode with `media_type_slug=all-media` is available and aggregate-only.
- Server mode with any other media type must be reported unavailable with an explicit reason.
- Per-media-type server saved views must not fall back to local saved lists.
- Unsupported reporting must include `collections.reading_list.per_media_type.server`.

## Saved Views And Saved Searches

Terminology:

- `Read-it-later saved view` means the media-type/context browser view for saved reading items.
- `Reading saved search` means a persisted query/filter record.

Contract:

- Local and server reading saved-search CRUD are supported where routed through `media.reading.saved_searches.*.<source>`.
- Server read-it-later saved browsing is constrained to the aggregate view.
- Per-media-type server saved views are unsupported with `server_contract_missing`.
- Invalid saved-context normalization may clean up presentation state but must not change the server contract.

## Chunk-Level TTS Adoption Decision

Decision: do not adopt chunk-level TTS as a first-class media/reading edge in this tranche.

Current supported behavior:

- Local reading TTS routes to the local TTS-capable reading service.
- Server reading TTS routes to the server reading TTS endpoint.
- Both are item-level operations keyed by reading item/media identity and source.

Deferred behavior:

- Per-chunk playback, per-chunk resume, chunk voice overrides, and chunk audio artifact adoption are deferred.
- Chunk-level TTS must not be faked by splitting item-level text in the scope service.
- A future chunk-level TTS contract must define chunk identity, transcript/content hash binding, artifact retention, progress/resume semantics, and source-specific deletion behavior before UI adoption.

Unsupported reason code for future reporting: `contract_deferred`.

## Unsupported Capabilities

Required unsupported reports:

| Operation ID | Source | Reason code | Contract |
|---|---|---|---|
| `collections.reading_list.per_media_type.server` | `server` | `server_contract_missing` | Server read-it-later browsing is aggregate-only. |
| `media.ingestion_sources.delete.server` | `server` | `server_contract_missing` | Server ingestion-source deletion is unavailable. |
| `media.reading.tts.chunk_level.server` | `server` | `contract_deferred` | Chunk-level server reading TTS adoption is deferred. |
| `media.reading.tts.chunk_level.local` | `local` | `contract_deferred` | Chunk-level local reading TTS adoption is deferred. |
| `media.web_content_ingest.local` | `local` | `source_specific_equivalent` | Local mode uses local URL ingest jobs instead of the direct server web-content ingest endpoint. |
| `media.transcription_models.local` | `local` | `source_specific_equivalent` | Local model discovery remains in local transcription settings. |
| `media.versions.advanced.local` | `local` | `local_contract_missing` | Advanced version rollback/metadata endpoints are server-owned; local mode has narrower helpers. |

The chunk-level TTS rows are contract rows for future unsupported reporting. Current scope-service reporting already covers the landed unsupported media rows and should add these rows only when a caller can attempt chunk-level TTS.

## Required Service Tests

Existing focused service tests are the required coverage:

- `tests/Media/test_media_reading_scope_service.py::test_scope_service_routes_server_ingest_jobs_and_reprocess_with_ingestion_job_actions`
- `tests/Media/test_media_reading_scope_service.py::test_scope_service_streams_server_ingest_job_events_with_observe_policy`
- `tests/Media/test_media_reading_scope_service.py::test_scope_service_read_it_later_context_capability_exposes_aggregate_metadata`
- `tests/Media/test_media_reading_scope_service.py::test_scope_service_reports_known_media_reading_capability_gaps`
- `tests/Media/test_media_reading_scope_service.py::test_scope_service_routes_server_reading_tts_with_policy`
- `tests/Media/test_media_reading_scope_service.py::test_scope_service_routes_local_reading_tts_with_policy`
- `tests/Media/test_media_reading_scope_service.py::test_scope_service_rejects_unsupported_server_ingestion_source_type_before_dispatch`
- `tests/Media/test_media_reading_scope_service.py::test_scope_service_server_ingestion_source_delete_enforces_policy_then_reports_unsupported`
- `tests/Media/test_server_media_ingest_jobs_service.py::test_server_media_service_routes_ingest_jobs_and_reprocess_operations`

No additional tests are required for this contract because the landed service tests already cover the required hard stops and source routing. The chunk-level TTS decision is documentation-only until an invokable chunk-level operation exists.
