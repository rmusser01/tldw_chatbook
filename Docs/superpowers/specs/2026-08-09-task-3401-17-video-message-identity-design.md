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
therefore assigns a different ID to video messages. On reload, the durable ID is
preserved in `ConsoleChatMessage.persisted_message_id`, while the Console tree
intentionally allocates a fresh native `message.id` for branch reconstruction.
Current video-card, play, and save-copy resolution incorrectly use that fresh native
ID and therefore cannot address retained bytes.

## Decision

Use the existing two identities for their intended jobs:

1. Reuse the stable-message-ID persistence seam. When a message carries
   `video_metadata`, `_persist_new_message()` passes `message.id` as the explicit
   `message_id` to `ChatPersistenceService.create_message()`, exactly as it already
   does for generation messages and callers using `force_stable_message_id`.
2. At every video-file resolution boundary, use
   `message.persisted_message_id or message.id`. The durable ID owns the store
   directory after persistence/restart; the native ID remains the fallback for a
   current, non-persisted message.
3. Restore the real Console action route by passing Play and Save-copy as named
   callables into `ConsoleMessageController`. The controller extraction currently
   dispatches to attributes it never received, so private screen-method tests alone
   would miss an `AttributeError` on the user-facing buttons.

No new identifier, metadata field, schema, file move, fallback scan, or migration
is introduced. Existing video rows remain valid tombstones; this change guarantees
identity only for newly persisted generated videos.

## Data Flow

1. The Console allocates one UUID before generation.
2. `run_video_generation()` saves bytes under that UUID.
3. `append_video_message(..., message_id=uuid, persist=True)` creates the live
   Console message with that UUID.
4. `_persist_new_message()` forwards that UUID to the existing persistence API.
5. On restart, the durable row restores that UUID as `persisted_message_id`, while
   its fresh native `id` remains available for Console tree navigation.
6. Card construction, playback, and save-copy resolve the file with
   `persisted_message_id` first and reconstruct the original path directly.

## Failure and Compatibility Behavior

- Explicit-ID conflicts continue to fail through the existing database conflict
  handling; the fix adds no retry or overwrite behavior.
- A synchronous persistence failure currently propagates after registering the live
  node and after bytes have been written. This task deliberately preserves that
  behavior: the existing ephemeral file remains subject to retention cleanup and no
  overwrite, rename, or speculative rollback is added. A focused conflict test pins
  propagation and proves an existing row is not replaced. Cleanup/transactional
  generation persistence would be a separate product behavior and is not required by
  this identity-repair task.
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
  the restored message's `persisted_message_id` and slug while confirming its native
  `id` is fresh.
- Focused card/action tests prove ready-state, playback, and save-copy use the durable
  key after reload rather than the fresh native ID. The Play and Save-copy tests
  enter through `handle_console_message_action()` so the production controller
  wiring is part of the proof.
- The new assertions are mutation-checked independently by removing the video
  stable-ID condition and by changing durable-first resolution back to `message.id`;
  each mutation must fail its corresponding focused test for the expected reason.
- A focused explicit-ID conflict test proves persistence failure propagates without
  replacing the existing row; no cleanup semantics are introduced.
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
