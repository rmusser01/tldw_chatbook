---
id: TASK-628
title: >-
  Lazy embeddings_rag dependency check never runs, Backfill dead for all new
  users
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 18:47'
updated_date: '2026-07-25 19:07'
labels:
  - rag
  - embeddings
  - bug
  - dependencies
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ensure_dependencies_checked() in tldw_chatbook/Utils/optional_deps.py has zero call sites anywhere in the codebase outside its own definition, so on the default lazy-checking configuration DEPENDENCIES_AVAILABLE['embeddings_rag'] stays at its initial False for the entire app lifetime. EmbeddingFactory (Embeddings_Lib.py:665) then refuses to initialize and RAG Backfill fails with a wrong could-not-be-created / install-dependencies error even when torch/transformers/chromadb/sentence_transformers are actually importable in the running environment. This breaks semantic RAG for effectively every new user out of the box.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With the default lazy dependency-check configuration, the gate that governs real embeddings/RAG service creation (EmbeddingFactory.__init__, the single choke point every EmbeddingsServiceWrapper / create_rag_service construction routes through) correctly reflects availability the first time it is actually reached, without requiring the eager-check env var or config flag
- [x] #2 Creating the shared RAG service / EmbeddingFactory for Backfill succeeds when the embeddings_rag optional packages are genuinely importable, with no eager import-time dependency checking added
- [x] #3 A test covering the lazy-default configuration demonstrates the embeddings_rag gate resolves to available when the underlying packages are importable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Root-cause confirm: grep every DEPENDENCIES_AVAILABLE['embeddings_rag'] / check_embeddings_rag_deps call site. Confirm ensure_dependencies_checked()/check_embeddings_rag_deps() have zero call sites under the default lazy mode, and trace the UAT log lines (embeddings_wrapper.py:296, ingestion_indexing.py:231) to EmbeddingFactory.__init__ (Embeddings_Lib.py:665), the single functional gate every EmbeddingFactory construction (including get_shared_rag_service -> simplified.create_rag_service -> EmbeddingsServiceWrapper) routes through.
2. RED: add Tests/RAG/test_lazy_embeddings_rag_dependency_check.py resetting the registry to its pristine never-checked state and asserting EmbeddingFactory construction + simplified.create_rag_service (the Backfill seam) both fail with the UAT's exact ImportError today. Confirm RED.
3. Fix at the root: add a small _embeddings_rag_available() helper beside EmbeddingFactory (reload-safe like the existing _current_dependencies_available()) that, only when the registry currently reads False, runs the real check_embeddings_rag_deps() probe once (a genuine "first use"), and use it in EmbeddingFactory.__init__'s gate. Keep laziness lazy: no import-time/module-load eager checking added anywhere.
4. Audit and update the one existing test whose simulation method (poking DEPENDENCIES_AVAILABLE directly) is now stale given the corrected behavior (Tests/Utils/test_optional_deps.py::test_embeddings_lib_graceful_failure) so it patches the underlying checker instead of only the flag.
5. Rerun the new tests (GREEN) plus Tests/RAG/ and Tests/RAG_Search/ in full for regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause confirmed exactly as suspected: ensure_dependencies_checked()/check_embeddings_rag_deps() have zero call sites anywhere the running app actually reaches under the default lazy mode (Tests/RAG/simplified/conftest.py even has a pre-existing comment: "workaround for the dependency check bug where it returns cached False"). DEPENDENCIES_AVAILABLE["embeddings_rag"] therefore sits at its pristine False default for the app's whole lifetime. The single functional consumer of that flag is EmbeddingFactory.__init__ (Embeddings_Lib.py) -- every EmbeddingsServiceWrapper / simplified.create_rag_service construction (including ingestion_indexing.get_shared_rag_service, the Backfill seam) routes through it, so this is the minimal covering fix point (not the search_rag_window.py UI-only display gates, which read the same flag for cosmetic purposes but were not part of the reported Backfill symptom and are left untouched here as a documented, separate follow-up).

Fix: added `_embeddings_rag_available()` beside the existing `_current_dependencies_available()` reload-safety helper in Embeddings_Lib.py. It trusts a True registry reading without re-probing, but on a False reading (which is always true on a fresh process under the default lazy mode) it runs the real `check_embeddings_rag_deps()` probe once, resolved via sys.modules the same reload-safe way `_current_dependencies_available()` already does. EmbeddingFactory.__init__'s gate now calls this instead of reading the raw flag. This keeps laziness lazy -- no import-time/eager checking was added anywhere -- while making the module's own "checked on first use" docstring claim literally true for this flag. A caller that already ran the real probe and got False (force_recheck_embeddings(), or a genuine missing-deps environment) is still honored, since the helper only substitutes a check for an UNCHECKED False, not a checked one.

RED/GREEN: added Tests/RAG/test_lazy_embeddings_rag_dependency_check.py (4 tests). Verified RED by git-stashing just the Embeddings_Lib.py change and confirming the two fix-verifying tests failed with the exact UAT ImportError message, while the pristine-state sanity check and the "explicit override still honored" test passed either way. Restored the fix and confirmed all 4 GREEN, including a direct EmbeddingFactory-gate test and an end-to-end simplified.create_rag_service (real, non-mock embedding model + in-memory vector store; the incidental network-touching embedding-dimension self-probe EnhancedRAGServiceV2 performs is monkeypatched out since it is unrelated to the dependency-gate bug being verified).

Found and fixed one pre-existing test whose simulation method the new behavior makes stale: Tests/Utils/test_optional_deps.py::test_embeddings_lib_graceful_failure previously simulated "missing deps" by poking DEPENDENCIES_AVAILABLE["embeddings_rag"] = False directly; since the real packages ARE importable in this environment, the new lazy re-check would have silently corrected that back to True. Updated it to also patch the underlying check_embeddings_rag_deps() checker itself, which is what genuinely simulates "the real probe ran and found the packages missing" post-fix.

Verification: Tests/RAG/test_lazy_embeddings_rag_dependency_check.py -> 4 passed. Tests/Utils/test_optional_deps.py -> 25 passed (was 24 passed/1 failed before the test update). Tests/RAG/ -> 562 passed, 8 skipped (baseline 558/8 + 4 new). Tests/RAG_Search/ -> 63 passed, 11 skipped.

Modified files:
- tldw_chatbook/Embeddings/Embeddings_Lib.py (_embeddings_rag_available() helper; EmbeddingFactory.__init__ gate now calls it)
- Tests/RAG/test_lazy_embeddings_rag_dependency_check.py (new, 4 tests)
- Tests/Utils/test_optional_deps.py (test_embeddings_lib_graceful_failure updated to patch the checker, not just the flag)

Known follow-up (not fixed here, out of this task's scope): tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py has two more raw DEPENDENCIES_AVAILABLE.get('embeddings_rag', False) reads (on_mount's missing-deps banner, and _start_indexing_run's guard) that share the same stale-registry anti-pattern and could show an incorrect "not available" UI on a fresh process before anything else has triggered a real check. Deliberately left alone: fixing them safely would require either an eager mount-time deep-import check (a real UX/perf regression -- torch/transformers import cost on the UI thread) or switching to the already-correct cheap embeddings_rag_deps_installed() find_spec probe used successfully elsewhere (semantic_availability.py, ingestion_indexing.semantic_indexing_available), which would also require updating two existing tests that currently patch the registry dict directly to simulate "missing". Worth a small dedicated follow-up task.
<!-- SECTION:NOTES:END -->
