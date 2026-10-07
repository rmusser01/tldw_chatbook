---
id: TASK-34430
title: Batch Chroma document deletes on reindex
status: Done
created_date: 2026-10-07 02:43
updated_date: 2026-10-07 21:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 6 / F17: re-ingest deletes stale chunks per document with a where scan per doc making bulk reindex O(N times C)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 5000 changed docs produce at most 10 delete calls,Feature detect falls back to per-doc loop on old Chroma,Tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 18 (T18)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ChromaVectorStore gained delete_documents(doc_ids): ids chunked into {"doc_id": {"$in": [...]}} where-deletes of at most _DOC_ID_DELETE_CHUNK_SIZE=500 each (5,000 docs -> exactly 10 calls, each list <= 500, all ids present; 501 -> [500, 1]). $in-on-delete support feature-detected ONCE per store (first call doubles as the probe, per-instance flag _batch_delete_supported): rejection logs one warning at detection time and every delete thereafter uses the per-doc loop (verified across two invocations: still exactly one warning). A batch call failing AFTER the probe succeeded (poison id) retries only its own 500 ids one-by-one, later chunks stay batched, feature detection is not reset; per-doc fallback failures log at debug (matching the old ingestion-path level) and never raise. Empty input -> zero calls; single doc -> one call with a one-id $in.

Caller (index_entries reindex path): collects the batch's changed ids and makes ONE delete_documents call per ingestion batch (no accumulation across batches); stores without the batch API (in-memory store, test fakes) keep the per-doc delete_document loop; a wholly-raising delete_documents is caught at debug and ingestion proceeds (best-effort preserved). delete_document itself left UNCHANGED rather than a thin wrapper -- remove_entries relies on its raise-through contract for orphan-reconciliation failure accounting; documented in its docstring. remove_entries and library_rechunk_service untouched.

Evidence (fake-collection call counts, 2,000-doc scenario): before 2,000 collection.delete calls (per-doc loop), after 4 (ceil(2000/500)) -- 500x fewer delete calls. Real chromadb 1.5.8 round-trip verified out-of-band (hermetic tests use a fake collection; no Chroma server/persisted client required): delete_documents removed exactly the targeted docs' chunks (18 -> 9), single-doc and empty inputs correct.

10 new hermetic tests in Tests/RAG_Search/test_batch_document_delete.py (TDD: red -> green). Targeted suites (test_ingestion_indexing, simplified/test_vector_store*, test_embedding_content_hash_cache, test_library_rechunk_service): 130 passed, failure list byte-identical to pre-change baseline (30 environmental: raw_participant bootstrap, rechunk status, selection env, integration-gated). ADR: not required (in-module perf optimization, no schema/contract/boundary change). Files: RAG_Search/simplified/vector_store.py, RAG_Search/ingestion_indexing.py, Tests/RAG_Search/test_batch_document_delete.py. Report: .superpowers/sdd/2026-10-06-nonconsole-efficiency-remediation/task-18-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
