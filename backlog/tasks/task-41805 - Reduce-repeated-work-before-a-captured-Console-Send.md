---
id: TASK-41805
title: Reduce repeated work before a captured Console Send
status: In Progress
created_date: 2026-11-19 01:49
assignee:
- '@codex'
labels:
- performance
- console
documentation:
- Docs/QA/console-library-responsiveness/README.md
updated_date: 2026-11-19 02:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce first-reply delay and UI-thread stalls when Capture On prepares and persists a long saved conversation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trace preparation numbers each domain in one pass and resolves every admitted canonical value exactly once with the existing ownership and value proofs.
- [x] #2 Persistence appends the same contiguous events and nodes inside the existing transaction, including omissions, rollbacks and the pre-provider dispatch checkpoint.
- [x] #3 Targeted trace and provider tests pass, and matching Terminal profiles measure preparation, worker persistence, loop gaps and resource retirement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record native reservation and dispatch worker spans. Pin ordinal work, canonical resolution count and append order with real SQLite regressions.
2. Assign ordinals in one pass, resolve values after domain selection and reuse the contiguous event sequence within the existing write.
3. Retest owner, continuation, replacement, omission, retry and rollback cases. Measure the same synthetic long Send in alternating Terminal runs and preserve all receipts.

ADR required: no
ADR path: N/A
Reason: ADR-337 already owns the transaction, trace alignment and provider entry rules. The work removes repeated operations inside those boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Trace preparation counts ordinals in one pass and resolves admitted values after selecting their domain. Direct saved messages use the existing bounded projector and reuse the revision returned by the owner check. The worker reads its append sequence once and continues to write the full ordered trace inside the existing transaction. ADR-337 applies. The 301 trace cases and 23 provider cases passed. Three alternating Terminal pairs preserved the ordered trace, Stop and retirement. Median reservation fell from 1.65 to 0.30 seconds, worker persistence from 2.41 to 2.27 seconds, and first streamed paint from 10.52 to 9.55 seconds. Two candidate busy frames exceeded 100 ms before reservation. The 50 ms loop limit and hosted verification remain open. The QA record retains measurements, the slower private readback control and the corrected pre-existing policy test.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
