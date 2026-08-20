---
id: TASK-19004
title: Add the private lasting-sync root registry
status: To Do
assignee: []
created_date: '2026-08-20 07:41'
labels:
  - notes
  - sync
  - database
dependencies:
  - TASK-19003
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the existing device-private notes.sync_state owner with local root, binding, representation, status, and receipt records while preserving the shipped import ledger contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The existing `notes.sync_state` private database migrates from the shipped import-ledger schema without losing or changing import receipt behavior.
- [ ] #2 Validated root, binding, representation, cursor, journal, recovery, migration, and bounded setting records support multiple local roots and illegal state transitions fail closed.
- [ ] #3 Active note scope, relative path, and stable file identity ownership are transactionally unique; candidate roots may not become active without a logical folder owner.
- [ ] #4 Public models and diagnostics expose opaque IDs and bounded reason codes only; paths, content, hashes, recovery bytes, and exception text stay private.
- [ ] #5 The owner remains backup- and export-excluded, is registered once in the private SQLite inventory, and all migration tests use isolated temporary databases.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: these accepted decisions already require the device-private registry, binding uniqueness, journal, recovery, and backup exclusion implemented here.
