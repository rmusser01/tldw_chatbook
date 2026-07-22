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
- [x] Vector collection names carry a versioned fingerprint over index-determining config only (embedding model + max_length, all chunking fields, distance_metric); query-time settings share one index.
- [x] Fingerprint inputs are normalized (TOML string and int hash identically) and the output is always a valid Chroma name.
- [x] Ingestion and search resolve to the same fingerprinted collection for a given active config (embed/query parity).
- [x] Existing pre-fingerprint `default` collections are adopted on first run under the active config's fingerprint — idempotent, race-safe, provenance-stamped as legacy/unverified; no existing index is silently blanked.
- [x] Collections carry provenance metadata (model, chunk params, fingerprint) stamped at creation.
- [x] A fresh fingerprinted collection reads as an honest empty index (reusing semantic_index_is_empty), distinct from zero-results.
- [x] Index admin API (list_indexes / delete_index / index_status) over the existing store CRUD.
- [x] Persistent-backend (Chroma) only for migration/empty-state; in-memory path unaffected.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-07-21-rag-index-isolation.md — 6 TDD tasks: (1) pure fingerprint module; (2) apply at the single store seam + stamp provenance; (3) legacy adoption migration; (4) index admin API; (5) empty-index honesty + parity; (6) docs. SDD-executed, user-gated PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SDD-executed (fresh subagent + task review per task, final opus whole-branch review = "ready to merge", no Critical/Important). 9 commits.

**New modules:** `RAG_Search/simplified/collection_fingerprint.py` (pure: `FINGERPRINT_VERSION`, `fingerprint_collection`, `fingerprinted_collection_name`, `collection_provenance`) and `collection_indexes.py` (Chroma-only: `adopt_legacy_collection`/`maybe_adopt_legacy_collection` migration + `list_indexes`/`delete_index`/`index_status`). **Seam:** `RAGService.__init__` resolves the collection via `fingerprinted_collection_name(self.config)` and stamps provenance — inherited by EnhancedRAGService/V2, so ingestion/backfill/search share the fingerprinted collection. Migration wired into `create_rag_service` (never blocks service creation).

**Fingerprint** covers index-determining fields only: `embedding.model`, `embedding.max_length`, all 10 `ChunkingConfig` fields, `vector_store.distance_metric`; excludes query-time (top_k/alpha/reranking/citations) and throughput fields. Versioned + normalized (str "400" == int 400), ASCII-only Chroma-safe names with no `..`.

**Two chromadb 1.5.8 landmines found + fixed during the migration:** (1) `collection.modify()` rejects the `hnsw:space` key even to restate it → strip it (which itself preserves the metric, held in `configuration_json`, not free-form metadata); (2) `SharedSystemClient` caches one client per persist_directory per process and errors on `Settings` mismatch → `collection_indexes._client()` matches `vector_store.py`'s `Settings` exactly; an e2e regression test locks it (verified by break-and-revert).

**Latent production bug fixed (surfaced by the empty-honesty contract test):** `ChromaVectorStore.get_collection_stats` raised numpy "truth value ambiguous" on `peek()["embeddings"]` (numpy arrays in 1.5.8), swallowed → masked the real count with `{count:0,error:...}` on essentially every call (had been degrading health_check→DEGRADED and get_chunk_count→0). Fixed with explicit `len()` checks; a positive-int `count` now flows to all consumers.

**Known limitation (documented):** only the canonical `default` legacy collection is auto-adopted; a user with a custom `[rag.vector_store].collection_name` gets a fresh empty fingerprinted collection and must re-index (out of scope per SP1 spec §4). **For SP2:** `create_config_for_collection` (`config.py`, used by `rag_search_tool.py`) is a second collection-name site that bypasses the migration, but it is search-only (never indexes) and was already disconnected from the real index pre-SP1 — no regression; flagged in case per-type collections get consolidated.

**Tests:** `Tests/RAG/simplified/{test_collection_fingerprint,test_collection_indexes,test_index_isolation_integration}.py`; full `Tests/RAG/` green (467→ with new tests, 8 pre-existing skips). Cross-SP invariant (adopted fingerprint == SP2 imported-profile fingerprint; embed==query config) is documented here and tested end-to-end in SP2.
<!-- SECTION:NOTES:END -->
