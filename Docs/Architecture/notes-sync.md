# Notes and bidirectional sync

This document describes the lasting file↔DB notes sync: the runtime owner, the review-first reconciler (this is **not** last-write-wins), the recovery-first executor, root leases, the watcher, and the device-state store. Session-scoped file-notes editing and guarded git publishing are covered at the end.

## Authoritative files

| File | Role |
| --- | --- |
| `Notes/notes_sync_runtime.py` | `NotesSyncRuntimeOwner` — app-owned runtime: `start`, `_reconcile`, `review_setup`, `check_root`, `request_sync_now`, `apply_reviewed`, `pause_root`/`resume_root`/`activate_root`, `shutdown` |
| `Notes/notes_sync_reconciler.py` | `plan_reconciliation()` — classifies immutable observations **without selecting winners** |
| `Notes/notes_sync_executor.py` | `NotesSyncExecutor` — durable, recovery-first execution of reviewed actions |
| `Notes/notes_sync_watcher.py` | `PollingNotesSyncWatcher` — polling hint emitter (1–10 s, jittered backoff) |
| `Notes/notes_sync_conflicts.py` | `NotesSyncConflictChoice` (KEEP_FILE / KEEP_NOTE / KEEP_BOTH / SKIP), bounded comparisons |
| `Notes/notes_sync_coordinator.py` | `NotesSyncRootCoordinator`, `RootLease` — per-root process lease; two Chatbook processes cannot own one root |
| `Notes/notes_device_state_store.py` | Durable roots/bindings/operations/recovery records at `<user_data_dir>/tldw_chatbook_notes_sync_state.db` |
| `Notes/notes_sync_authority.py` + `notes_scope_service.py` | `NotesScopeSyncAuthority` (DB-side CRUD authority), global/workspace scopes |
| `Notes/sync_paths.py` + `notes_sync_filesystem.py` | Descriptor-anchored FS boundary (`O_NOFOLLOW`, dir-fd pinned ops, 10 MiB file cap, xattr bounds) |
| `Notes/notes_sync_legacy.py` | Read-only migrator of legacy evidence into paused candidates — the only place LWW ever existed |
| `Notes/file_notes_service.py` | `FileNotesService` — session-scoped file-notes editing (disk authority) |
| `Notes/file_notes_git_*.py`, `git_process_containment.py` | Guarded session git commit/push (ADR-038/039) |

There is no single `Notes/sync_engine.py` — that name is legacy; the "lasting sync" engine is the module set above.

## The model: review-first, not LWW

Synced extensions are `.md`, `.markdown`, `.txt`. The reconciler classifies each binding from immutable observations:

| Observation | Plan |
| --- | --- |
| File changed, note unchanged | `UPDATE_NOTE` (disk → DB) |
| Note changed, file unchanged | `UPDATE_FILE` (DB → disk) |
| **Both changed** | **CONFLICT** (`both_sides_changed`) — surfaced for explicit resolution, never auto-picked |
| One side missing | `DELETION_REVIEW` |
| Direction violation | `out_of_direction_*` conflict |
| `duplicate_authority` | PAUSE the root |
| Move with identity mismatch | `ambiguous_identity` conflict |
| Root offline / overlapping / capability loss / stale observation | Plan-level skip with reason |

Last-write-wins (`newer_wins` and friends) exists only in the legacy policy vocabulary consumed by the read-only migrator.

## Dataflow (edit → sync)

1. The polling watcher detects a changed root and emits a scheduling hint.
2. The runtime observes the root: file and note observations against device-state baselines, plus an observation token.
3. `plan_reconciliation` classifies bindings into safe actions and attention items (conflicts, deletion reviews, pauses) — no winners chosen.
4. The UI shows the reviewed plan; `apply_reviewed` accepts only the **current** token (a stale plan requires "Check again" — content may have moved).
5. `NotesSyncExecutor.execute` applies exactly the reviewed actions with a staged durable journal; a mid-apply crash leaves a resumable incomplete operation (`_resume_incomplete` classifies and finishes it on next start).
6. Conflicts resolve explicitly KEEP_FILE / KEEP_NOTE / KEEP_BOTH; `undo_resolution` reverses via linked operation ids; retained conflict copies live for 30 days.

## Root coordination

Each synced folder is a **root** with a process lease: admission validates reparse points and overlap; two Chatbook processes cannot own one root simultaneously. Roots can be paused/resumed/activated explicitly; duplicate-authority conditions pause rather than guess.

## Config keys

`notes.recovery_capacity_bytes` (default 256 MiB; env `TLDW_NOTES_SYNC_RECOVERY_CAPACITY_BYTES`), `notes.sync_watcher_interval_seconds` (1.0), `notes.sync_watcher_max_interval_seconds` (10.0, ceiling 3600).

## Session file-notes and git publishing

Separately from lasting sync, `FileNotesService` provides session-scoped editing of file-backed notes (open/save/move/delete/restore/reconcile) with **disk authority** (ADR-029): the file is the truth; the DB follows. Guarded git publishing (`file_notes_git_*`) performs session commits and pushes with process containment (ADR-035/038/039) — commit and push are separately gated user actions, never automatic.

Note templates (`Config_Files/create_custom_template.py`, stored in the user config dir) support `{date}`/`{time}`/`{datetime}` placeholders and feed the Library Notes UI.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Both sides changed | Bounded machine-readable CONFLICT attention; explicit resolution required |
| Mid-apply crash | Durable journal; resumable incomplete op on next start |
| Stale review plan | Token mismatch; re-check required before apply |
| Root not available / overlapping | PAUSE with reason code and next-action copy |
| Oversized file / xattr bounds | Refused at the FS boundary |
| Second process claims a root | Lease refusal |

## Governing decisions

ADR-021 (file-backed notes disk authority and recovery), ADR-029 (file-notes disk authority), ADR-035 (session git index controls), ADR-038/039 (guarded session commit/push), ADR-059 (notes folder import and device-local sync ownership), ADR-073 (round-trip and interoperability constraints). Feature doc: `Docs/Features/notes_bidirectional_sync.md` (owner table matches code).

## Verified gotchas

1. "Bidirectional LWW sync" in old docs is wrong for the current engine — both-sides-changed is always a reviewed conflict.
2. The watcher is polling-only (no FS events dependency); hints drive reconciliation timing.
3. Executor worker coroutines run one loop per worker thread — an earlier per-call `asyncio.run` leaked SQLite connections.
4. `duplicate_authority` pauses the root rather than picking a winner, because two bindings claiming one note cannot be reconciled safely.
