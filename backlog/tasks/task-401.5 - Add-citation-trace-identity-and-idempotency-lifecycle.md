---
id: TASK-401.5
title: Add citation trace identity and idempotency lifecycle
status: To Do
assignee: []
created_date: '2026-07-24 00:44'
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
