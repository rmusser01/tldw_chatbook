---
id: TASK-640
title: >-
  Bundle three task-641/635 review Minors: build-lock-free config resolution,
  discarded-build cleanup, real-config integration test
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 21:55'
updated_date: '2026-07-26 03:47'
labels:
  - followup
  - uat
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-641/635 review (three bundled Minors): (1) resolve_active_rag_config() is currently evaluated inside get_shared_rag_service()'s _shared_service_lock/_shared_service_build_lock section before the blocking create_rag_service() call -- it does its own config-profile disk reads and could itself be moved further out to shrink the window builders hold the build lock. (2) A build discarded because a concurrent reset superseded its generation (ingestion_indexing.py's _shared_service_generation check) is currently just dropped/left for GC -- no explicit cleanup call is made on it, which is fine today (EnhancedRAGServiceV2 has no close()/shutdown() contract) but should be revisited if the underlying service ever gains real resources (open file handles, network clients) that need releasing. (3) task-635's _has_legacy_rag_config_material() predicate is only exercised today against monkeypatched get_cli_setting fakes in Tests/RAG/test_first_run_import.py -- an integration test reading a REAL on-disk config.toml (with and without a legacy [AppRAGSearchConfig.rag] section) would close the gap between the unit-level fakes and the real TOML-parsing path.

Additional finding (task-641 round-2 live re-UAT, noted but not fixed there): the builtin `hybrid_basic` RAG profile's embedding model id `all-MiniLM-L6-v2` 404s against the Hugging Face Hub -- it is missing the `sentence-transformers/` org prefix (the canonical id is `sentence-transformers/all-MiniLM-L6-v2`). This is currently caught and gracefully defaulted to dim=768 (see rag_service.py's `_get_embedding_dimension`/`RAGService.__init__`), so it does not block indexing/search, but it means the builtin profile's actual semantic/embedding path silently degrades (wrong/no real embedding model ever loads) for every fresh install that uses the default profile without ever explicitly configuring a different model.

task-641 round-3 review Minors (protect_file_descriptors(), the fd-ownership fix in Embeddings_Lib.py/transcription_service.py/higgs.py): (4) protect_file_descriptors() mutates GLOBAL process state (sys.stdout/sys.stderr/os.environ) with no lock guarding the whole context-manager body -- two threads calling it concurrently (e.g. two worker threads both loading HuggingFace models at once) can race on these reassignments; worth a lock around the full body (or documenting why it's accepted as out-of-scope for a genuinely single-first-caller-wins use case). (5) The function is duplicated verbatim in 3 files (Embeddings_Lib.py, transcription_service.py, higgs.py; chatterbox.py imports its copy from Embeddings_Lib.py) -- should be consolidated into one shared utility all three import, so a future fix only needs to land once instead of being copy-pasted 3x (as both the closefd=False fix and the created_out/created_err fix just were). (6) Test coverage gap: existing tests cover sys.stdout.fileno() raising AttributeError (no fileno() method); add a variant where fileno() returns -1 successfully (some stream implementations do this instead of raising) to confirm the downstream os.fstat(-1) -> OSError path is also exercised and still correctly triggers the except-branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 resolve_active_rag_config() (or an equivalent minimal-config-only read) is evaluated with the smallest practical critical section under _shared_service_build_lock -- documented if left as-is with rationale
- [x] #2 Discarded-build cleanup is either implemented or explicitly deferred with a code comment/task noting EnhancedRAGServiceV2 has no close()/shutdown() contract today
- [x] #3 An integration test exercises _has_legacy_rag_config_material() against a real on-disk config.toml for both the legacy-present and fresh-install cases
- [x] #4 protect_file_descriptors() either gains a lock guarding its full context-manager body, or the decision to leave it unguarded is explicitly documented with rationale
- [x] #5 The 3 verbatim copies of protect_file_descriptors() (Embeddings_Lib.py, transcription_service.py, higgs.py) are consolidated into one shared implementation, or consolidation is explicitly deferred with rationale
- [x] #6 Test coverage includes a fileno() returns -1 (not raises) variant for the non-fd-backed-stream detection path
- [x] #7 Builtin profiles' bare embedding model id all-MiniLM-L6-v2 no longer 404s against the HF Hub at the point the embedding loader resolves model_name_or_path, without changing the embedding.model fingerprint string any existing profile/collection was built under (proven by a test pinning the affected builtin profile's collection_fingerprint.fingerprint_collection() output)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Item 4 (HF model id) investigated first (may reshape scope): confirmed EmbeddingsServiceWrapper._build_config() in RAG_Search/simplified/embeddings_wrapper.py is the sole embedding-loading layer (self.model_name flows straight from RAGConfig.embedding.model, the exact fingerprint field collection_fingerprint.py's _index_fields hashes) -- fix is a narrow bare-id-to-canonical-HF-id alias map applied ONLY to the model_name_or_path handed to the HF loader, never to model_name/self.model_name. Golden fingerprint for the hybrid_basic builtin profile pinned before the change (bf912f0fe11b) to prove stability after.
2. Item 6 first (test infra), then item 5 (consolidation lands the lock once): create tldw_chatbook/Utils/fd_protection.py with the consolidated protect_file_descriptors() (byte-equal union of the 3 existing copies -- transcription_service.py/higgs.py's harmless subprocess.Popen save/restore no-op included since dropping it would diverge from 2 of 3 sites) plus a module-level threading.Lock() guarding the FULL context-manager body (setup through yield through finally, per item 4's actual race concern). Repoint Embeddings_Lib.py/transcription_service.py/higgs.py to import protect_file_descriptors from the shared module (re-export so existing test import paths keep working unchanged); repoint chatterbox.py's 3 dynamic imports from Embeddings_Lib to the new shared module directly. Remove now-unused `from contextlib import contextmanager` (all 3) and `import subprocess` (higgs.py only). New consolidated test file Tests/Utils/test_fd_protection.py ports the existing Tests/Embeddings/test_embeddings_lib_protect_file_descriptors.py coverage plus: a fileno()->-1 (not raising) variant (item 6), a lock-held-for-full-body test (item 4), and a same-object-across-all-4-import-sites identity test (regression guard for item 5). Delete the superseded Tests/Embeddings/test_embeddings_lib_protect_file_descriptors.py.
3. Item 1: restructure get_shared_rag_service() (RAG_Search/ingestion_indexing.py) so resolve_active_rag_config()/_configured_profile() run BEFORE _shared_service_build_lock is acquired at all (not just before _shared_service_lock) -- redundant concurrent config reads are cheap/side-effect-free and the existing generation check at swap time already discards a build superseded by a concurrent reset, so this is a safe, spurious-discard-worst-case change per the task-641 review. Preserve the top fast-path None-check, the first-run-import-before-any-lock ordering, and re-check _shared_service/generation inside the lock exactly as before.
4. Item 2: implement discarded-build cleanup in get_shared_rag_service()'s two discard branches. Investigation found EnhancedRAGServiceV2 (via EnhancedRAGService/RAGService) DOES already define a real close() (shuts down its thread pool executor, embeddings, vector store, DB connection pools) -- contrary to the task description's premise -- so the honest fix is to call it (defensively via getattr/callable-check for forward safety) on a discarded build, always OUTSIDE _shared_service_lock (close() can block) so a slow close never reintroduces the task-641 fast-lock hazard.
5. Item 3: add a real on-disk config.toml integration test for _has_legacy_rag_config_material() in Tests/RAG/test_first_run_import.py -- no monkeypatched get_cli_setting, using the autouse isolate_test_environment fixture's already-isolated TLDW_CONFIG_PATH/HOME, writing real TOML content with and without a legacy [AppRAGSearchConfig.rag.*] section before calling the function directly.
6. Run pytest Tests/RAG/ -q, Tests/Embeddings/ Tests/Transcription/ -q, the new fd test file, and Tests/TTS/ (chatterbox/higgs import-path smoke) as gates; one commit per logical item (or sensible grouping), no push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all 7 ACs (6 bundled minors from the task-641/635 review, plus AC#7 added for the HF model id finding that had no AC yet).

1/2 (ingestion_indexing.py): resolve_active_rag_config()/_configured_profile() now run BEFORE _shared_service_build_lock is acquired at all (previously inside both locks), shrinking the fast-lock window; the existing generation check at swap time still guards correctness (worst case a safe spurious discard). Added _close_discarded_rag_service(): investigation found EnhancedRAGServiceV2 (via EnhancedRAGService/RAGService) already has a real close() (thread pool, embeddings, vector store, DB pools) contrary to the task's premise -- now called (getattr/callable-guarded) on both discard branches, always outside _shared_service_lock so a slow close() can't reintroduce the task-641 hazard. 3 new tests (2 discard-closes-the-build, 1 proving the new ordering), verified RED against the pre-fix code via git stash before landing GREEN.

3 (active_config.py): added 3 real on-disk config.toml integration tests for _has_legacy_rag_config_material() in Tests/RAG/test_first_run_import.py -- no monkeypatched get_cli_setting, using the autouse isolate_test_environment fixture's per-test TLDW_CONFIG_PATH.

4/AC#7 (embeddings_wrapper.py): chose strategy (a) from the task brief -- normalize the bare "all-MiniLM-L6-v2" builtin model id ONLY at the point EmbeddingsServiceWrapper._build_config() hands model_name_or_path to the HF loader, via a narrow explicit alias map (_BARE_HF_MODEL_ID_ALIASES), never touching model_name/RAGConfig.embedding.model (the fingerprint field). Fingerprint stability proven by a test pinning the real hybrid_basic builtin profile's fingerprint_collection() output (bf912f0fe11b, computed independently before the change and asserted equal after).

5/6 (protect_file_descriptors consolidation): new tldw_chatbook/Utils/fd_protection.py; discovered transcription_service.py/higgs.py carried a vestigial (no-op) subprocess.Popen save/restore that Embeddings_Lib.py's copy lacked -- kept it in the consolidated version (harmless, avoids diverging from 2/3 sites). Embeddings_Lib.py/transcription_service.py re-export under the old name (existing test imports unchanged); chatterbox.py's 3 dynamic imports repointed directly to the new module. Added a module-level threading.Lock() guarding the FULL context-manager body (setup through yield through finally) for AC#4. Tests/Utils/test_fd_protection.py supersedes Tests/Embeddings/test_embeddings_lib_protect_file_descriptors.py: ports all prior coverage plus a fileno()->-1 variant (AC#6), a lock-held-for-full-body test, and a same-object-across-all-4-import-sites identity guard. Cleaned up now-dead `from contextlib import contextmanager` (3 files) / `import subprocess` (higgs.py) / `import os` (Embeddings_Lib.py, shadowed elsewhere) / `import sys` (higgs.py), pyflakes-verified clean.

Gates: Tests/RAG/ 583 passed/8 skipped; Tests/Embeddings/+Tests/Transcription/+Tests/Utils/test_fd_protection.py 131 passed/38 skipped; Tests/TTS/ 807 passed/14 skipped; Tests/RAG_Search/ 55 passed. No regressions.

Modified/added files: tldw_chatbook/RAG_Search/ingestion_indexing.py, tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py, tldw_chatbook/Utils/fd_protection.py (new), tldw_chatbook/Embeddings/Embeddings_Lib.py, tldw_chatbook/Local_Ingestion/transcription_service.py, tldw_chatbook/TTS/backends/higgs.py, tldw_chatbook/TTS/backends/chatterbox.py, Tests/RAG/test_ingestion_indexing.py, Tests/RAG/test_first_run_import.py, Tests/RAG/simplified/test_collection_fingerprint.py, Tests/RAG_Search/test_embeddings_unit.py, Tests/Utils/test_fd_protection.py (new), Tests/Embeddings/test_embeddings_lib_protect_file_descriptors.py (deleted, superseded).
<!-- SECTION:NOTES:END -->
