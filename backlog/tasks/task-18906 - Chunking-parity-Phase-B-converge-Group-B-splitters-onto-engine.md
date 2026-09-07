---
id: TASK-18906
title: 'Chunking parity Phase B: converge Group B splitters onto the engine'
status: Done
assignee: []
created_date: '2026-08-19 09:30'
updated_date: '2026-08-19 09:30'
labels:
  - chunking
dependencies:
  - TASK-18905
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase B (PR 2) of the Chunking Engine Parity sub-project: delete the regex splitter `_chunk_text_in_process` and route `ChunkingService.chunk_text` through the engine for all methods (validation messages preserved, flat per-chunk contract preserved, method whitelist dropped so `ebook_chapters` works through the RAG service entry point); retire `EnhancedChunkingService` in favor of the engine's `structure_aware` + a parent/child adapter that keeps the `chunk_with_parent_retrieval` shape for the two RAG-indexing hot paths and the preview modal; converge `local_media_reading_service._chunk_text` (the raw char slicer) onto the engine; verify preview/ingest agreement.

Plan: `Docs/superpowers/plans/2026-08-19-chunking-engine-parity.md` Tasks 7–10.
Spec: `Docs/superpowers/specs/2026-08-18-chunking-engine-parity-design.md` (§6.3, §7.2, §7.3; rulings Q5/Q6).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `_chunk_text_in_process` is deleted; `ChunkingService.chunk_text` routes all methods through the engine; the three validation errors ("max_words must be positive", "Overlap must be non-negative", "Overlap must be less than max_words") still raise `ChunkingError` with identical messages
- [x] #2 `ebook_chapters` returns chapter chunks through the `RAG_Search.chunking_service` entry point instead of raising `InvalidChunkingMethodError` (spec §7.2 regression fixed)
- [x] #3 `EnhancedChunkingService`'s home-grown structure/hierarchy logic is deleted; a parent/child adapter over the engine's hierarchical output preserves the `chunk_with_parent_retrieval` return shape (`chunks` + `parent_chunks` with parent references) for `enhanced_indexing_helpers.py`, `enhanced_rag_service.py`, and the preview modal
- [x] #4 `local_media_reading_service._chunk_text` delegates to the engine; no mid-word splits; `perform_chunking=False` still returns `[]`
- [x] #5 Preview modal and ingestion path produce identical chunk texts for the same input and options (spec §7.3), with the preview's word-count column non-zero
- [x] #6 All chunk output keeps the flat per-chunk contract (top-level `text`/`start_char`/`end_char`/`word_count`); DB round-trip characterization still passes
- [x] #7 Targeted suites green (`Tests/Chunking/`, `Tests/RAG/`, `Tests/Media/test_local_media_chunking.py`, import-weight guard)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Delete regex splitter, route all methods through engine, preserve validation + flat contract (plan Task 7)
2. Parent/child adapter + ECS retirement + consumer re-pointing (plan Task 8)
3. Converge the char slicer in `local_media_reading_service` (plan Task 9)
4. Preview/ingest agreement test + Phase B close-out (plan Task 10)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Commits `bf75eee20..d30d62d57`: deleted the regex splitter and routed `ChunkingService.chunk_text` fully through the engine (validation messages and flat per-chunk contract preserved, `ebook_chapters` unblocked), retired `EnhancedChunkingService` in favour of the engine's `structure_aware` plus a parent/child adapter preserving the `chunk_with_parent_retrieval` shape for the two RAG-indexing hot paths and the preview modal, converged `local_media_reading_service._chunk_text` onto the engine, and added the preview/ingest agreement test.

- **Latent shift**: the old char slicer's chunk sizes were char-based; the engine's words strategy measures in words, so raw-`_chunk_text` call sites see a units change (latent, not user-visible in chunk quality terms) — flagged as the known chars→words shift.
- Full detail: `.superpowers` reports; ADR `backlog/decisions/071-vendored-chunking-engine-parity.md`.
