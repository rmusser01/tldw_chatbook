# ADR-229: Time-based retention window for the local sync journals

Date: 2026-10-11
Status: Accepted
Task: [TASK-34389](../tasks/task-34389%20-%20Define-and-wire-a-sync-log-retention-policy.md)

## Context

Both local stores keep a `sync_log` change journal. ChaChaNotes_DB writes it
from AFTER INSERT/UPDATE/DELETE triggers on nine entities, and each row's
`payload` is the complete entity row as JSON — including `content` for every
message. `Client_Media_DB_v2._log_sync_event` does the same for media and
chunk events. The result is a second full copy of every mutated body, plus
the live row itself: ~2x write amplification and O(total mutations) growth.

Retention APIs existed but nothing called them: `prune_sync_log`
(reachability-based, added with the task-19564 v45 triggers), and
`delete_sync_log_entries*` on both stores. The only production reader is
`Prompt_Management/Prompts_Interop.py`, which pages the journal by
`change_id` cursor with a limit. The ChaChaNotes reachability triggers bound
that journal to the frontier its committed-intent readers can reach, but the
frontier rows of live entities still live forever, and the media journal has
no bound at all.

## Decision

1. **90-day default retention on the local sync journal**, enforced by age:
   a single parameterized `DELETE FROM sync_log WHERE timestamp < ?` per
   store, over the indexed ISO-8601 `timestamp` column. New methods
   `delete_sync_log_entries_older_than(cutoff)` on `ChaChaNotes_DB` and
   `Client_Media_DB_v2`; the app computes the cutoff from the window.
2. **Config-visible via `[sync] sync_log_retention_days`** in the default
   config TOML (default `90`). `0` (or any value `<= 0`) disables pruning
   entirely. A malformed non-integer value logs a warning and skips the
   pass — it never disables the default silently in the other direction.
3. **Runs on the existing post-startup maintenance path**, not the staggered
   boot-worker fleet: `_perform_sync_log_retention` is scheduled from
   `schedule_media_cleanup` (which `_schedule_deferred_startup_work` calls
   after first paint) — once ~65 s after startup, then every 24 h. All DB
   work runs on a worker thread (`asyncio.to_thread`); the pass is wrapped
   so no failure can surface to the UI or block boot. Pruned-row counts are
   logged.
4. **Payload slimming is explicitly deferred** pending a consumer audit:
   payloads keep carrying full bodies. The journal stays a self-contained
   sync proof; shrinking it to a content hash is a format change to a live
   proof and needs every consumer enumerated first (the committed-intent
   readers compare payloads field-by-field).
5. **Readers are unaffected within the window.** The production reader pages
   by `change_id` cursor; pruning only removes cold history. The Console
   committed-intent readers (`read_committed_chat_sync_intent` and its
   callers) look up exact `(message_id, version)` rows; their write-time and
   restore-time uses touch recent rows, and the outbox producer already
   treats a missing source intent as a benign
   `{"status": "skipped", "reason": "source_intent_unavailable"}`.

## Alternatives considered

- **Wire the existing reachability prune into startup** (`prune_sync_log`):
  already enforced incrementally by the v45 triggers on ChaChaNotes, and it
  does not exist for the media journal. It bounds reachability, not age —
  the frontier copy of every live entity still accumulates forever.
- **Delete-by-change_id watermark after a consumer acks**: needs a real
  sync server to ack against; none exists in this product today. Premature.
- **Payload slimming now** (store a content hash instead of the body):
  the larger win on write amplification, but it changes the payload format
  that field-by-field intent readers validate. Deferred until the consumers
  are audited; this ADR keeps the journal correct and bounded first.
- **Fixed policy, no config**: rejected — the window governs how long a
  second copy of user text is retained locally; that is a user-visible
  data-retention choice and belongs in config.

## Security or data consequences

- The journal is a second copy of conversation/note/media **content**;
  before this decision that copy survived indefinitely even for entities
  deleted or purged elsewhere. A 90-day window bounds how long deleted
  content remains recoverable from `sync_log` (the ChaChaNotes reachability
  triggers already purge unreachable bodies eagerly; this bounds the rest).
- Pruning is deletion of local history: once a row is pruned, a future
  full-state sync cannot be reconstructed from this client's journal alone.
  Within the default window nothing is removed, and `0` restores the old
  unbounded behavior for users who need the journal intact.
- The ChaChaNotes tombstone (`delete`) rows are content-free and age out
  with everything else under the window; the reachability layer's guarantees
  for live readers are unchanged, and delete proofs older than the window
  are no longer retained.
- No content leaves the device: pruning is a local DELETE, parameterized,
  with no identifiers interpolated.

## Verification

- `Tests/DB/test_sync_log_retention.py` — both stores' new methods delete
  only rows strictly older than the cutoff (rows written through the real
  trigger / `_log_sync_event` paths included), keep the row stamped exactly
  at the cutoff, keep emitting after a prune; the app pass prunes both
  stores, `0` disables, a missing store and a malformed value are contained;
  the shipped config template, the defaults tree, and the real loader all
  carry the knob (legacy configs without `[sync]` resolve the default 90).
- `Tests/Architecture/test_timer_path_static_update_inventory.py` — the new
  24 h interval clock is registered in `EXPECTED_CLOCK_ROOTS`.
- Targeted runs of the pre-existing sync-log suites
  (`Tests/DB/test_chachanotes_sync_log_retention*.py`,
  `Tests/Media_DB/test_media_db_v2.py -k sync_log`) and the config-defaults
  suites pass; the retention change is additive and touches no reader.
