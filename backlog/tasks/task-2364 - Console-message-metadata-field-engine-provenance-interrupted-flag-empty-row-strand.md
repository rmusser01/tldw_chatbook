---
id: TASK-2364
title: >-
  Console message metadata field: engine provenance, interrupted flag, empty-row
  strand
status: Done
assignee: []
created_date: '2026-08-04'
updated_date: '2026-08-05 05:04'
labels:
  - console
  - realtime
  - store
dependencies: []
priority: medium
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: model tests for a frozen MessageMetadata (closed transcript-status vocabulary, JSON round trip, corrupt/legacy degrade).
2. RED: v30->v31 migration tests (column added, version bumped, excluded from messages_sync_* payloads, idempotent when pre-applied, local-only write leaves version/last_modified/sync_log untouched).
3. GREEN: MessageMetadata + messages.metadata_json column, mirroring the local-only usage_json precedent end to end (DDL file + runner guard, add_message/update_message/SELECTs, update_message_metadata_local).
4. GREEN: plumbing -- ChatPersistenceService kwargs + update_message_metadata, ConsoleChatMessage.metadata, store append/set/persist paths, resume read-back (tree walk + screen-state snapshot).
5. Consumers: realtime rows carry engine/provider/model; interrupted rides the flag (reseed builder reads it, legacy content-marker fallback kept for pre-field rows); empty/failed input transcripts record a transcript status instead of stranding an empty row.
6. Docs: replace the V4 spec's Continuity deferral note with what shipped; tick ACs + Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `MessageMetadata` (`tldw_chatbook/Chat/message_metadata.py`) -- a frozen
dataclass carrying `engine`/`provider`/`model`, `interrupted` and
`transcript_status` -- persisted as the local-only `messages.metadata_json`
column, ChaChaNotes schema **v30 -> v31**.

**Approach.** The `usage_json` column (v29 -> v30, cost ticker) was followed
seam for seam, because the two columns have identical properties: both record
what THIS device observed, neither can ever ride a sync payload. So:
`migrations/chachanotes_v30_to_v31_message_metadata.sql` stays a plain
unconditional ALTER while `_migrate_from_v30_to_v31` owns the idempotence guard
(`PRAGMA table_info` skips only the DDL, never the version bump); no
`messages_sync_*` trigger is touched; and `update_message_metadata_local`
provides a version-neutral write so a metadata-only flush cannot trip the
update trigger's `WHEN` clause and enqueue a sync row that could not carry the
column anyway. Plumbing mirrors the same precedent: optional `metadata_json`
kwargs (omitted, never NULL, so a content-only write cannot clobber),
`ChatPersistenceService.update_message_metadata`, `append_message(metadata=...)`
and `ConsoleChatStore.set_message_metadata`, plus read-back on both round trips
(conversation resume and screen-state snapshots).

**Decisions and trade-offs.**
- *Frozen dataclass with a CLOSED `transcript_status` vocabulary*, refused at
  construction. A dict would have moved the guessing from content parsing to key
  spelling. `from_json` still degrades (unknown keys dropped, unknown status ->
  `""`, corrupt payload -> `None`) because it runs against durable data on the
  resume path, where raising is never the right answer.
- *The visible `⏹ interrupted` marker stays in the content.* Users need it, and
  exports render content verbatim -- moving it out would have silently dropped
  interruption from every export. What changed is that the reseed builder
  DECIDES from `metadata.interrupted` and only then trims the suffix it wrote;
  a turn whose words merely contain that text is no longer mangled. Rows written
  before v31 have no flag, so the old unconditional strip is kept for exactly
  those (dropping it would have restarted marker-seeding on older conversations).
- *Exports/summaries needed no conversion*: grepping the marker and the constant
  found only the reseed builder. Recorded here so the next reader does not go
  looking.
- *`transcript_status` is written AFTER the content write*, never before -- a row
  claiming `final` whose text never landed is precisely the class of lie this
  field exists to remove. The empty case (`on_input_transcript` with no words)
  now records `empty` on the row it used to abandon; a failed write records
  `failed`.
- *An adopted pipeline capture claims no transcription model* (`model=""`): its
  words came from local STT, not from the realtime provider.

**Verification.** In-memory/`tmp_path` databases only. The schema bump was
deliberately never exercised against the real `~/.local/share/tldw_cli` database:
every other worktree on this machine is still at v30 and the DB layer refuses,
by design, to open a database newer than the code. That hazard is now recorded
in `backlog/docs/lessons-live-verification.md`.

**Modified or added files.** Added `Chat/message_metadata.py`,
`DB/migrations/chachanotes_v30_to_v31_message_metadata.sql`,
`Tests/Chat/test_message_metadata.py`,
`Tests/DB/test_chachanotes_message_metadata_migration.py`. Modified
`DB/ChaChaNotes_DB.py`, `Chat/console_chat_models.py`,
`Chat/console_chat_store.py`, `Chat/chat_persistence_service.py`,
`Chat/chat_conversation_service.py`, `UI/Screens/chat_screen.py`, the V4 spec's
Continuity section, `backlog/docs/lessons-live-verification.md`, and the
store/resume/realtime test suites (plus four schema-version literals in sibling
DB tests that move with each migration).
<!-- SECTION:NOTES:END -->

## Description (the why)

`ConsoleChatMessage` has no metadata field, so the V4 spec's engine provenance
(engine/provider/model) and interruption marking ride a visible " ⏹ interrupted" content
marker plus usage-attach (documented deferral, spec Continuity section). Consequences: the
marker is fed back to the model on reseed only via a strip hack; exports/summaries
string-match UI copy; a legitimately-empty transcript strands an empty user row forever
with nothing recording why. A store-level metadata field closes all three.

## Acceptance Criteria (the what)

- [x] Messages can carry structured metadata (engine, provider, model, interrupted,
      transcript-status) without content markers.
- [x] The reseed builder and exports read the field instead of string-matching.
- [x] The spec's Continuity deferral note is updated to point at the shipped field.
