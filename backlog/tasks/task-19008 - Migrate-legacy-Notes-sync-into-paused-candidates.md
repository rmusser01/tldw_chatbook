---
id: TASK-19008
title: Migrate legacy Notes sync into paused candidates
status: To Do
assignee: []
created_date: '2026-08-20 07:50'
labels:
  - notes
  - sync
  - migration
dependencies:
  - TASK-19004
  - TASK-19005
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Translate legacy root and per-note sync evidence into reviewable paused lasting-root candidates without reading absence as deletion or mutating notes files or legacy history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Distinct safe legacy roots become separate paused candidates with recognizable bindings, bounded reports, and a source fingerprint.
- [ ] #2 Config-only, row-only, missing, overlapping, duplicate, out-of-root, unsafe, and invalid evidence is represented honestly without inferring deletion.
- [ ] #3 Migration writes only the private device store and never creates folders, managed memberships, notes, files, receipts, watchers, or active roots.
- [ ] #4 Legacy conflict winners and auto-sync settings are historical evidence only and never become lasting-root policy.
- [ ] #5 Repeated migration is idempotent, preserves legacy history read-only, and every candidate requires a complete current dry-run plus explicit activation.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: ADR-059 already mandates mutation-free paused-candidate migration and forbids running both owners.
