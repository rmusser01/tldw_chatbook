---
id: TASK-19007
title: Execute lasting sync through a durable recovery journal
status: To Do
assignee: []
created_date: '2026-08-20 07:45'
labels:
  - notes
  - sync
  - recovery
dependencies:
  - TASK-19004
  - TASK-19005
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Admit recovery capacity before destructive work and execute guarded local note and filesystem operations through resumable durable journal states with verified outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local note and managed-membership reads and writes pass only through `NotesScopeService`; the executor never opens ChaChaNotes or File Notes authority directly.
- [ ] #2 Recovery capacity and durable intent are admitted before destructive work, and pending, unresolved, or Undo-eligible recovery cannot be evicted.
- [ ] #3 Each operation revalidates observations, advances a durable stage after each authority mutation, verifies both outcomes, updates binding ownership, and completes last.
- [ ] #4 Interruption resumes only against matching observations; stale or partial outcomes become explicit attention with bounded resume, restore, or disconnect choices.
- [ ] #5 Capacity failure, cancellation, and injected failure after every stage produce no blind replay, false atomicity, or hidden mutation.
- [ ] #6 Logs and public diagnostics exclude content, paths, hashes, recovery bytes, credentials, and raw exception text.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/027-portable-database-note-session-coordinator.md`, `backlog/decisions/055-library-destructive-action-reversibility-rule.md`, `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: these decisions already fix service ownership, recovery admission, journaling order, and interruption semantics.
