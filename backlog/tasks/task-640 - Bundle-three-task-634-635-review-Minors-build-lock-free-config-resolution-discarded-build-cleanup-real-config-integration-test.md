---
id: TASK-640
title: >-
  Bundle three task-634/635 review Minors: build-lock-free config resolution,
  discarded-build cleanup, real-config integration test
status: To Do
assignee: []
created_date: '2026-07-25 21:55'
updated_date: '2026-07-26 00:02'
labels:
  - followup
  - uat
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-634/635 review (three bundled Minors): (1) resolve_active_rag_config() is currently evaluated inside get_shared_rag_service()'s _shared_service_lock/_shared_service_build_lock section before the blocking create_rag_service() call -- it does its own config-profile disk reads and could itself be moved further out to shrink the window builders hold the build lock. (2) A build discarded because a concurrent reset superseded its generation (ingestion_indexing.py's _shared_service_generation check) is currently just dropped/left for GC -- no explicit cleanup call is made on it, which is fine today (EnhancedRAGServiceV2 has no close()/shutdown() contract) but should be revisited if the underlying service ever gains real resources (open file handles, network clients) that need releasing. (3) task-635's _has_legacy_rag_config_material() predicate is only exercised today against monkeypatched get_cli_setting fakes in Tests/RAG/test_first_run_import.py -- an integration test reading a REAL on-disk config.toml (with and without a legacy [AppRAGSearchConfig.rag] section) would close the gap between the unit-level fakes and the real TOML-parsing path.

Additional finding (task-634 round-2 live re-UAT, noted but not fixed there): the builtin `hybrid_basic` RAG profile's embedding model id `all-MiniLM-L6-v2` 404s against the Hugging Face Hub -- it is missing the `sentence-transformers/` org prefix (the canonical id is `sentence-transformers/all-MiniLM-L6-v2`). This is currently caught and gracefully defaulted to dim=768 (see rag_service.py's `_get_embedding_dimension`/`RAGService.__init__`), so it does not block indexing/search, but it means the builtin profile's actual semantic/embedding path silently degrades (wrong/no real embedding model ever loads) for every fresh install that uses the default profile without ever explicitly configuring a different model.

task-634 round-3 review Minors (protect_file_descriptors(), the fd-ownership fix in Embeddings_Lib.py/transcription_service.py/higgs.py): (4) protect_file_descriptors() mutates GLOBAL process state (sys.stdout/sys.stderr/os.environ) with no lock guarding the whole context-manager body -- two threads calling it concurrently (e.g. two worker threads both loading HuggingFace models at once) can race on these reassignments; worth a lock around the full body (or documenting why it's accepted as out-of-scope for a genuinely single-first-caller-wins use case). (5) The function is duplicated verbatim in 3 files (Embeddings_Lib.py, transcription_service.py, higgs.py; chatterbox.py imports its copy from Embeddings_Lib.py) -- should be consolidated into one shared utility all three import, so a future fix only needs to land once instead of being copy-pasted 3x (as both the closefd=False fix and the created_out/created_err fix just were). (6) Test coverage gap: existing tests cover sys.stdout.fileno() raising AttributeError (no fileno() method); add a variant where fileno() returns -1 successfully (some stream implementations do this instead of raising) to confirm the downstream os.fstat(-1) -> OSError path is also exercised and still correctly triggers the except-branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 resolve_active_rag_config() (or an equivalent minimal-config-only read) is evaluated with the smallest practical critical section under _shared_service_build_lock -- documented if left as-is with rationale
- [ ] #2 Discarded-build cleanup is either implemented or explicitly deferred with a code comment/task noting EnhancedRAGServiceV2 has no close()/shutdown() contract today
- [ ] #3 An integration test exercises _has_legacy_rag_config_material() against a real on-disk config.toml for both the legacy-present and fresh-install cases
- [ ] #4 protect_file_descriptors() either gains a lock guarding its full context-manager body, or the decision to leave it unguarded is explicitly documented with rationale
- [ ] #5 The 3 verbatim copies of protect_file_descriptors() (Embeddings_Lib.py, transcription_service.py, higgs.py) are consolidated into one shared implementation, or consolidation is explicitly deferred with rationale
- [ ] #6 Test coverage includes a fileno() returns -1 (not raises) variant for the non-fd-backed-stream detection path
<!-- AC:END -->
