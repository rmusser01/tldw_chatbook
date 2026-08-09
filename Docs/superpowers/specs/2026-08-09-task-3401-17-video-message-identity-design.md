# TASK-3401.17 Generated Video Message Identity Design

## Goal

Keep one stable identifier across the Console message, its durable database row,
and the `VideoStore` directory so TTL-retained generated video bytes resolve after
an app restart.

## Context

ADR-044 stores generated video bytes at
`generated_videos/<message_id>/<slug>.mp4` and persists only the video's marker
and metadata in the message row. The Console preallocates a UUID before generation
and writes the bytes under that UUID. `ConsoleChatStore.append_video_message()`
uses the same UUID for the live message, but `_persist_new_message()` currently
requests an explicit database ID only for image-generation messages. The database
therefore assigns a different ID to video messages. After reload, the durable ID
becomes the restored Console message ID and no longer addresses the existing file.

## Decision

Reuse the existing stable-message-ID persistence seam. When a message carries
`video_metadata`, `_persist_new_message()` will pass `message.id` as the explicit
`message_id` to `ChatPersistenceService.create_message()`, exactly as it already
does for generation messages and callers using `force_stable_message_id`.

No new identifier, metadata field, schema, file move, fallback scan, or migration
is introduced. Existing video rows remain valid tombstones; this change guarantees
identity only for newly persisted generated videos.

## Data Flow

1. The Console allocates one UUID before generation.
2. `run_video_generation()` saves bytes under that UUID.
3. `append_video_message(..., message_id=uuid, persist=True)` creates the live
   Console message with that UUID.
4. `_persist_new_message()` forwards that UUID to the existing persistence API.
5. On restart, the durable row restores with the same UUID and `VideoStore.resolve()`
   reconstructs the original path directly.

## Failure and Compatibility Behavior

- Explicit-ID conflicts continue to fail through the existing database conflict
  handling; the fix adds no retry or overwrite behavior.
- Persistence failures retain the current deferred/error behavior and do not rename
  or duplicate video bytes.
- Session retention remains unchanged and is owned by TASK-3401.19.
- Preview/player behavior remains unchanged and is owned by TASK-3401.18.
- Existing rows whose IDs already diverge are not repaired because the storage key
  is intentionally absent from durable metadata and the UAT scratch media was removed.

## Verification

- A focused store-level test proves a persisted video message forwards its live ID
  and receives the same persisted ID.
- A real SQLite integration test saves bytes under a preallocated ID, persists the
  video message, reconstructs the Console store from the durable conversation tree,
  creates a fresh TTL-configured `VideoStore`, and resolves the retained bytes using
  the restored message ID and slug.
- The new assertion is mutation-checked by removing the video stable-ID condition and
  confirming the focused test fails for the expected identity mismatch.
- Only test files that exercise the touched persistence/store paths are run.

## Alternatives Rejected

- Persisting a second storage-key field duplicates identity and changes the durable
  metadata contract without need.
- Renaming the file after database insertion adds rollback and orphan-file failure
  modes to a flow that already owns a stable UUID before generation.
- Scanning store directories by slug is ambiguous and abandons ADR-044's direct
  message-keyed resolution contract.

## ADR Check

ADR required: no

ADR path: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

Reason: ADR-044 already requires message-keyed generated-video storage. This task
corrects the implementation to satisfy that existing boundary without changing the
storage, persistence, or lifecycle contract.
