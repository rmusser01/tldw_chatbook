---
id: TASK-484
title: Fix builtin RAG profiles with invalid chunking_method values
status: Done
assignee: []
created_date: '2026-07-22 01:45'
updated_date: '2026-07-24 15:03'
labels:
  - rag
  - profiles
  - followup
dependencies:
  - task-483
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from SP2a (task-483) Task 2 review. Three builtin profiles in `tldw_chatbook/RAG_Search/config_profiles.py` set the CORRECT field `chunking_method` but to values that are NOT in the pipeline's valid set (words/sentences/paragraphs/tokens/semantic/json/ebook_chapters/xml/rolling_summarize): `hybrid_full` and `research_papers` use `"hierarchical"`, `technical_docs` uses `"structural"`. These are silently unused today, but if the enhanced-chunking code path (`Chunker.chunk_text`) is ever exercised for a profile with these values it raises `InvalidChunkingMethodError`. This is a different bug class from the SP2a dead-attribute fix (correct field, invalid value), so it was left as a follow-up.

Decide the correct valid `chunking_method` for each (these builtins also set `preserve_structure`; "paragraphs" is the closest structure-respecting valid method) and set it, or remove the line to fall back to the default. Needs a small design decision on intended chunking behavior for each.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every builtin profile's `chunking_method` is a value accepted by the runtime chunker (no `InvalidChunkingMethodError` possible from any builtin).
- [x] #2 A test asserts all builtins use a valid `chunking_method`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified runtime behavior before fixing (empirically, not just by reading):
- Chunk_Lib.Chunker.chunk_text's dispatch (Chunking/Chunk_Lib.py ~line 622-705) only
  accepts words/sentences/paragraphs/tokens/semantic/json/ebook_chapters/xml/
  rolling_summarize; anything else hits the terminal `else` and raises
  InvalidChunkingMethodError.
- "Simple" path (RAG_Search/chunking_service.py ChunkingService.chunk_text, used by
  RAGService._chunk_document -- the default indexing flow and the enhanced-service
  fallback when parent-retrieval structural chunking isn't used): does NOT silently
  fall back. Only "words"/"sentences"/"paragraphs" get the in-process fast path
  (line 61); any other value (including "hierarchical"/"structural") falls through
  to constructing a Chunker and calling Chunker.chunk_text(method=...), which raises
  Chunk_Lib.InvalidChunkingMethodError. ChunkingService.chunk_text's own broad
  except re-wraps that into its own (distinct) ChunkingService.ChunkingError.
  Confirmed live: chunk_text(method="hierarchical") raises
  "Failed to chunk text using method 'hierarchical': Unsupported chunking method:
  'hierarchical'".
- Higher callers (RAGService.index_document, chunk_documents_batch) catch that
  exception per-document and return IndexingResult(success=False, chunks_created=0,
  error=...) rather than crashing the app -- so end-to-end the failure mode is
  "every document silently indexes to zero chunks / errors out" rather than a hard
  crash, but it is 100% reachable, not latent, whenever the simple/fallback
  chunking path runs for one of these three builtins.
- EnhancedChunkingService.chunk_text_with_structure (enhanced_chunking_service.py)
  DOES have its own dispatch that explicitly accepts "hierarchical"/"structural"
  and does not raise -- but its only caller, chunk_with_parent_retrieval, hardcodes
  method="structural" and never reads config.chunking_method at all, so that branch
  never actually exercises the profile's configured value either way (a separate,
  pre-existing dead-setting issue, out of scope here).
- Net: the profile's chunking_method value IS read and DOES raise
  (InvalidChunkingMethodError, wrapped) on the default/fallback indexing path -- the
  bug was real and reachable, just manifesting as a caught per-document indexing
  failure rather than an app crash.

Fix: set hybrid_full, technical_docs, and research_papers' chunking_method to
"paragraphs" (from "hierarchical"/"structural") -- the closest runtime-valid,
structure-respecting method, matching each profile's existing preserve_structure=True
intent. Left an inline comment on each line noting the prior (invalid) value and why.

Fingerprint consequence: chunking_method is one of the fields hashed into the
collection fingerprint (simplified/collection_fingerprint.py), so these three
builtins' fingerprints change with this fix. Any collection previously built while
cloned/seeded from the old (invalid) value will no longer match and will re-point to
a freshly built collection on next use. This is expected/correct for a config-value
bug fix in seed profiles -- those old fingerprints were never reachable via a working
enhanced-chunking run anyway.

Test: added test_all_builtins_use_a_runtime_valid_chunking_method to
Tests/RAG/test_config_profiles.py, asserting every builtin's chunking_method is in
a RUNTIME_VALID_CHUNKING_METHODS set enumerated (with a comment pointing at the
source) from Chunk_Lib.Chunker.chunk_text's elif dispatch chain. RED before the fix
(failed on exactly hybrid_full/technical_docs/research_papers), GREEN after.

Regression: Tests/RAG/ -- 537 passed, 8 skipped (pre-existing skips, unrelated).
Tests/UI/test_settings_rag_profile_adapter.py -- 55 passed.

Files touched: tldw_chatbook/RAG_Search/config_profiles.py,
Tests/RAG/test_config_profiles.py.
<!-- SECTION:NOTES:END -->
