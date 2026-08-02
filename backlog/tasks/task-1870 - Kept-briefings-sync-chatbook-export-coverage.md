---
id: TASK-1870
title: 'Kept briefings: sync/chatbook-export coverage'
status: To Do
assignee: []
created_date: '2026-08-02 00:16'
labels:
  - watchlists
  - briefings
  - chachanotes
  - persistence
  - sync
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up filed at task-1780's close-out, per that spec's "Non-goals (v1)" section
(`Docs/superpowers/specs/2026-08-01-kept-briefings-design.md`).

Task-1780 added `kept_briefings`/`kept_scripts` to ChaChaNotes (schema v29,
`tldw_chatbook/DB/migrations/chachanotes_v28_to_v29_kept_briefings.sql`) so that generated
briefings/scripts a user chooses to keep survive watchlist deletion and Subscriptions_DB
pruning. This was a deliberate, recorded v1 decision (spec, "Schema" section): the two tables
carry **no sync columns** (`client_id`/`version`/`deleted`) — they do not participate in
ChaChaNotes's existing bidirectional sync machinery, and deletion is a hard `DELETE`, not the
soft-delete-flag convention every synced entity in this DB uses.

The same gap exists on the chatbook-export side: chatbook export already knows how to walk
conversations, notes, characters, and other ChaChaNotes entities into a portable bundle, but has
no awareness of `kept_briefings`/`kept_scripts` at all — a user's kept briefings and cast scripts
are silently absent from any chatbook they export today.

Whether closing this gap means adding sync columns and wiring the tables into the existing sync
engine, adding a dedicated (non-sync) export path in the chatbook exporter, both, or neither with
a recorded rationale for staying out (e.g. "kept content is meant to be local-only for now") is
an open design question — not decided here. What matters is that the gap stops being silent:
either these artifacts become reachable through at least one of the two systems users already
rely on for taking their data with them, or the decision to leave them out is written down
somewhere a future reader will find it before assuming coverage that doesn't exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A user's kept briefings and their kept scripts are included when the user exports a chatbook, OR a recorded decision explains why they are deliberately excluded
- [ ] #2 A user's kept briefings and their kept scripts participate in ChaChaNotes sync between devices, OR a recorded decision explains why they are deliberately excluded
- [ ] #3 Whatever is decided for #1 and #2 is written down (spec, ADR, or equivalent) so the next reader does not have to reverse-engineer it from the absence of code
<!-- AC:END -->
