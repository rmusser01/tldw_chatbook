---
id: TASK-2364
title: >-
  Console message metadata field: engine provenance, interrupted flag, empty-row
  strand
status: In Progress
assignee: []
created_date: '2026-08-04'
updated_date: '2026-08-05 04:40'
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
- [ ] The reseed builder and exports read the field instead of string-matching.
- [ ] The spec's Continuity deferral note is updated to point at the shipped field.
