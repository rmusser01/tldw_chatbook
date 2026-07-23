---
id: TASK-495
title: >-
  Merge legacy query-time RAG keys into the first-run "Imported settings" profile
status: To Do
assignee: []
created_date: '2026-07-23 02:00'
updated_date: '2026-07-23 02:00'
labels:
  - rag
  - profiles
  - followup
dependencies:
  - task-487
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from SP2b (task-487, PR for config-resolution unification) final review. `ensure_imported_profile` snapshots the active-profile resolution (built-in `hybrid_basic` + env on a true first run), which is correct for the SP1 index-fingerprint invariant (pre-SP2b ingestion built the collection from the built-in profile, so SP1 adopted under the built-in fingerprint). But it does NOT capture a user's hand-tuned NON-fingerprint-affecting query-time legacy keys (`[AppRAGSearchConfig.rag.search].default_top_k`, `score_threshold`, `include_citations`, reranking settings). Those are silently discarded on import.

Enrich the first-run snapshot: merge such hand-set query-time legacy keys onto the built-in base (they do not affect the SP1 fingerprint, so the index invariant is unaffected — only merge NON-index-determining fields to keep the fingerprint equal to SP1's adoption). Do NOT merge embedding-model / chunk fields from legacy keys (that would change the fingerprint and orphan the legacy collection).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The "Imported settings" first-run profile preserves a user's hand-set query-time legacy keys (top_k, score_threshold, citations, reranking) from `[AppRAGSearchConfig.rag.*]`.
- [ ] The imported profile's SP1 fingerprint still equals SP1's adopted legacy-collection fingerprint (only non-index-determining fields merged) — a test asserts this holds with legacy query-keys set.
<!-- AC:END -->
