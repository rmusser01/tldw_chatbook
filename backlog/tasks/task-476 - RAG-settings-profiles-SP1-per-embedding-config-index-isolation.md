---
id: TASK-476
title: >-
  RAG settings + profiles SP1: per-embedding-config index isolation
status: In Progress
assignee: []
created_date: '2026-07-21 16:30'
updated_date: '2026-07-21 16:30'
labels:
  - rag
  - profiles
  - index
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sub-project 1 of 3 in the RAG settings screen + profiles program (overview spec: Docs/superpowers/specs/2026-07-21-rag-settings-profiles-overview.md; SP1 spec: Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md; plan: Docs/superpowers/plans/2026-07-21-rag-index-isolation.md). Makes vector collections keyed by the config that built them so that changing embedding model, chunking, or distance metric points at a distinct index instead of silently corrupting a shared one. Prerequisite for SP2 (profile system) and SP3 (settings screen).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Vector collection names carry a versioned fingerprint over index-determining config only (embedding model + max_length, all chunking fields, distance_metric); query-time settings share one index.
- [ ] Fingerprint inputs are normalized (TOML string and int hash identically) and the output is always a valid Chroma name.
- [ ] Ingestion and search resolve to the same fingerprinted collection for a given active config (embed/query parity).
- [ ] Existing pre-fingerprint `default` collections are adopted on first run under the active config's fingerprint — idempotent, race-safe, provenance-stamped as legacy/unverified; no existing index is silently blanked.
- [ ] Collections carry provenance metadata (model, chunk params, fingerprint) stamped at creation.
- [ ] A fresh fingerprinted collection reads as an honest empty index (reusing semantic_index_is_empty), distinct from zero-results.
- [ ] Index admin API (list_indexes / delete_index / index_status) over the existing store CRUD.
- [ ] Persistent-backend (Chroma) only for migration/empty-state; in-memory path unaffected.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-07-21-rag-index-isolation.md — 6 TDD tasks: (1) pure fingerprint module; (2) apply at the single store seam + stamp provenance; (3) legacy adoption migration; (4) index admin API; (5) empty-index honesty + parity; (6) docs. SDD-executed, user-gated PR.
<!-- SECTION:PLAN:END -->
