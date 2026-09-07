---
id: TASK-553.5
title: Add citation trace identity and idempotency lifecycle
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 09:40'
labels:
  - rag
  - citations
  - identity
  - reliability
dependencies:
  - TASK-553.4
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make persistence retries, cache reuse, and message edits attach or invalidate the correct immutable trace without duplication, while preserving the dormant import and Sync namespace contracts for later transports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local, server, imported, payload, and owner identities follow the ADR-024 namespace rules and enforce uniqueness.
- [x] #2 An uncertain message-plus-trace persistence retry is idempotent and cannot create partial or duplicate aggregate rows.
- [x] #3 Cache hits add owners to the original trace instead of cloning or renaming it.
- [x] #4 A body-fingerprint mismatch or unavailable fingerprint key removes active grounded presentation while retaining aggregate-only historical provenance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing namespaced idempotency-domain tests for local retry, server wire identity, owner links, cache reuse, imported, and dormant Sync constructors.
2. Add uncertain-commit retry tests covering one message/aggregate/child/owner and fail-closed identity reuse with different body or governed payload.
3. Implement stable codec-derived repository idempotency comparisons and cache-owner reuse without cloning or rewriting generation identity.
4. Add active owner-body fingerprint lookup and ChatPersistenceService edit invalidation, preserving historical aggregate reads while preventing unverified grounded presentation.
5. Run focused identity/repository/service plus adjacent persistence/migration/benchmark regressions, lint, diff, and both independent review gates.
6. Complete acceptance criteria and implementation notes only after approval.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This task implements ADR-024’s accepted identity, idempotency, cache-owner, and body-binding lifecycle using the existing schema and codec contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented six domain-separated HMAC identity constructors for local retry, authenticated server wire identity, message-owner links, cache-owner reuse, imported traces, and dormant Sync operations. Namespace inputs include complete profile/origin/authority/tenant/wire/external identity, remain bounded, and never serialize raw text, secrets, or portable digests.

Repository persistence is idempotent across uncertain committed-result retries: exact messages, aggregates, child rows, attachments, feedback, and owners are reused only after immutable identity/integrity comparison; conflicting body, trace, governed payload, authority, selected-answer HMAC, or owner identity fails closed without partial mutation. Cache hits add one owner to the original trace only after verifying the exact persisted candidate message and the available selected-answer body/HMAC; they never clone or rename the trace or generation.

Active grounded presentation is available only through repository-issued, weakref/digest-registered capabilities. Selected answer text and MESSAGE_BODY HMAC are bound through preparation, the live message row, stable identity, and execution before citation inserts. Verification rechecks current identity, exact message revision/body, active owner fingerprint/state, trace visibility, and keyed integrity; edits, deletion, owner changes, visibility changes, identity drift, tampering, wrong repositories, and stale cache payloads invalidate and unregister the capability while historical summaries remain readable. ChatPersistenceService edit transitions and same-body revision carry-forward are atomic; mismatch or unavailable key never remains grounded.

Verification: 356 adjacent tests passed; final focused quality review passed 157 tests; Ruff check/format, compileall, and git diff checks passed. Qualification remained eligible with overall_pass=true: sealed write p95 5.480 ms, summary read p95 0.226 ms, and 35/35 trace-owner rows. Independent specification and quality/security reviews approved the final implementation with no remaining Critical or Important findings. The benchmark fixture was updated only to derive the exact selected-body HMAC required by the hardened contract; the committed baseline remained unchanged.

ADR required: yes. Applied existing backlog/decisions/024-rag-citation-provenance-and-source-resolution.md; no new ADR was needed.
<!-- SECTION:NOTES:END -->
