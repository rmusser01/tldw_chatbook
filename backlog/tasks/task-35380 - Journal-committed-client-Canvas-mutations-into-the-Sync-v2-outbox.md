---
id: TASK-35380
title: Journal committed client Canvas mutations into the Sync v2 outbox
status: To Do
created_date: 2026-10-18 05:56
dependencies:
- TASK-35376
labels:
- canvas
- sync
- client
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Capture durable eligible Canvas changes without losing or duplicating source across crashes between the artifact database and sync-state database. Reuse existing source-intent/outbox receipt ownership under ADR-239.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Only committed durable Canvas mutations belonging to explicitly synchronized conversations produce immutable source intents in the owning message/artifact transaction; cancellation, rollback and Temporary sessions enqueue nothing.
- [ ] #2 Exact source snapshots, origins, scope and stable envelope IDs reconcile into the existing Sync v2 outbox with idempotent source receipts; crash/retry does not reconstruct data from a later selected revision or require a distributed transaction.
- [ ] #3 Identity/server/account/dataset changes fence stale intents, incoming no-echo writes do not re-enqueue, and hard-purge cleanup covers source-bearing intents and queued copies.
- [ ] #4 Targeted tests cover direct rename, assistant settlement, atomic promotion, rollback, enqueue/reconcile crashes and source-free diagnostics; outbound Canvas transport remains disabled until separately negotiated integration qualifies.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
