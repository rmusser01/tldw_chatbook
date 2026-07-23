---
id: TASK-487
title: >-
  RAG settings + profiles SP2b: config-resolution unification onto the active profile
status: To Do
assignee: []
created_date: '2026-07-23 00:00'
updated_date: '2026-07-23 00:00'
labels:
  - rag
  - profiles
dependencies:
  - task-483
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second half of SP2 (profile system) in the RAG settings screen + profiles program (overview: Docs/superpowers/specs/2026-07-21-rag-settings-profiles-overview.md; SP2 spec: Docs/superpowers/specs/2026-07-21-rag-profile-system-design.md; plan: Docs/superpowers/plans/2026-07-22-rag-profile-system-sp2b-config-unification.md). Makes the active profile the single source the RAG engine reads, so ingestion and search never diverge. Depends on SP2a (task-483, PR #780 merged) and SP1 (task-476, PR #771 merged). SP3 (settings screen) follows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] One `resolve_active_rag_config()` (active profile's rag_config deep-copy + env override layer) is the single config source, consumed by both the search path (`RAGConfig.from_settings` delegates to it) and the ingestion path (shared service built from it).
- [ ] Parity: for a given active profile + env, the config used to embed (ingestion) equals the config used to query (search) — fingerprint-equal (anti dimension-crash); locked by a test.
- [ ] The single active pointer `[rag.service].profile` is reused (no second key); `set_active_profile()` writes it and resets the shared service (via the existing `reset_shared_rag_service()`).
- [ ] `fusion.py` hybrid_alpha resolves from the active profile; scattered `AppRAGSearchConfig.rag.*` value reads deprecated (the `rag.indexing` toggle read is left as-is).
- [ ] First-run import: the resolved config is snapshotted into an "Imported settings" active profile whose SP1 fingerprint equals the SP1 adopted-legacy fingerprint (cross-SP invariant, tested e2e) — no silent index blanking on upgrade.
- [ ] No DB migration. Shared-service singleton reset is test-isolated (task-408).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-07-22-rag-profile-system-sp2b-config-unification.md — 6 TDD tasks: (1) resolve_active_rag_config + env-overlay extraction (from_settings delegates); (2) shared service via the resolver (parity); (3) fusion hybrid_alpha from active profile; (4) set_active_profile (pointer + reset); (5) first-run "Imported settings" import + fingerprint invariant; (6) deprecation cleanup + docs + regression gate. New module active_config.py. SDD-executed, user-gated PR. NOTE: reset_shared_rag_service() ALREADY EXISTS (ingestion_indexing.py:192).
<!-- SECTION:PLAN:END -->
