---
id: TASK-401.5
title: Add citation trace identity and idempotency lifecycle
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 08:38'
labels:
  - rag
  - citations
  - identity
  - reliability
dependencies:
  - TASK-401.4
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-401
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make persistence retries, cache reuse, and message edits attach or invalidate the correct immutable trace without duplication, while preserving the dormant import and Sync namespace contracts for later transports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local, server, imported, payload, and owner identities follow the ADR-024 namespace rules and enforce uniqueness.
- [ ] #2 An uncertain message-plus-trace persistence retry is idempotent and cannot create partial or duplicate aggregate rows.
- [ ] #3 Cache hits add owners to the original trace instead of cloning or renaming it.
- [ ] #4 A body-fingerprint mismatch or unavailable fingerprint key removes active grounded presentation while retaining aggregate-only historical provenance.
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
