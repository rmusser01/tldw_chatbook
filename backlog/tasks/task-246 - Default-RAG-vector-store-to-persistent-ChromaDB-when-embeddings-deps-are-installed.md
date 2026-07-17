---
id: TASK-246
title: >-
  Default RAG vector store to persistent ChromaDB when embeddings deps are
  installed
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:11'
labels:
  - rag
  - embeddings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The RAG vector store defaults to type memory (RAG_Search/simplified/config.py:48) and the hybrid_basic runtime profile does not override it, so the semantic index starts empty on every launch and nothing can persist across sessions. tldw_server uses persistent ChromaDB as its default store. When the embeddings_rag extra is installed, the chatbook RAG service should default to a persistent ChromaDB store under the user data directory so indexed content survives restarts. Filed from the 2026-07-12 RAG module audit (see backlog/docs/rag-module-audit-2026-07-12.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With embeddings_rag installed the RAG service vector store persists across app restarts (ChromaDB with a persist directory under the user data dir)
- [x] #2 Without embeddings deps behavior is unchanged: plain FTS5 search works and no import errors occur
- [x] #3 Config still allows an explicit override back to the in-memory store
- [x] #4 Store selection logic is covered by tests for both with-deps and without-deps cases
<!-- AC:END -->

## Implementation Plan

1. Add store-selection helpers to `tldw_chatbook/RAG_Search/simplified/config.py`:
   - `_embeddings_rag_available()` — memoized wrapper around
     `Utils.optional_deps.check_embeddings_rag_deps()` (the 'embeddings_rag' feature key),
     returning False on any failure so environments without the extras never error.
   - `default_vector_store_type()` — resolution order: `RAG_VECTOR_STORE` env var >
     explicit `[AppRAGSearchConfig.rag.vector_store].type` in user config > "chroma"
     when embeddings_rag deps are installed > "memory" fallback.
   - `default_chroma_persist_directory()` — `RAG_PERSIST_DIR` env var > explicit
     `persist_directory` in user config > `get_user_data_dir() / "chromadb"` (same
     user-data-dir convention as the ChaChaNotes/Media DB paths).
2. Change `VectorStoreConfig` to resolve its defaults in `__post_init__`: the `type`
   field defaults to an "auto" sentinel that resolves via `default_vector_store_type()`,
   and a `None` `persist_directory` is auto-filled for chroma stores. Explicitly passed
   values (constructor args, `from_dict`, TOML via `from_settings`) always win, so an
   explicit `type = "memory"` keeps today's behavior. Because runtime profiles
   (`config_profiles.py`, e.g. `hybrid_basic`) build plain `RAGConfig()` objects, they
   inherit the new default with no profile changes — keeping the diff off the files a
   parallel agent is editing.
3. TDD: new `Tests/RAG/simplified/test_vector_store_selection.py` covering with-deps
   default (chroma + persist dir under user data dir), without-deps default (memory,
   no chromadb import required), explicit memory override via config and env var,
   explicit persist-directory override, `from_settings` behavior, `hybrid_basic`
   profile inheritance, and a chroma persistence round-trip (importorskip-gated).
4. Run targeted suites (`Tests/RAG/`, new tests, related smoke tests) and verify no
   regressions against the recorded baseline (239 passed / 8 skipped).

## Implementation Notes

Implemented as planned — all changes live in `RAG_Search/simplified/config.py` plus tests;
`config_profiles.py`, `rag_factory.py`, and `vector_store.py` are untouched.

- `VectorStoreConfig.type` now defaults to an `"auto"` sentinel resolved in
  `__post_init__` via `default_vector_store_type()`: `RAG_VECTOR_STORE` env var >
  explicit `[AppRAGSearchConfig.rag.vector_store].type` > `"chroma"` when
  `optional_deps.check_embeddings_rag_deps()` (feature key 'embeddings_rag') passes >
  `"memory"`. A `None` persist_directory is auto-filled for chroma stores via
  `default_chroma_persist_directory()` (`RAG_PERSIST_DIR` > explicit config >
  `get_user_data_dir()/"chromadb"`, matching the app DB path convention). The deps
  probe is memoized (`_EMBEDDINGS_RAG_AVAILABLE`) and fails closed to memory.
- Resolving in the dataclass constructor (rather than in profiles or the factory) means
  every construction path — runtime profiles like `hybrid_basic` (plain `RAGConfig()`),
  `from_settings()`, `from_dict()` — inherits the persistent default, while explicitly
  passed values always win, so an explicit `type = "memory"` in user config now wins on
  the profile path too (it previously wasn't consulted there at all).
- Side benefit: the deps probe populates the lazy `DEPENDENCIES_AVAILABLE` registry, so
  `Tests/test_smoke.py::test_rag_service_initialization` now genuinely initializes the
  service instead of skipping on a false "dependencies not installed" ImportError
  (registry was never populated in a bare process). That smoke test got a tmp_path
  persist dir to stay hermetic under the new default.
- New tests: `Tests/RAG/simplified/test_vector_store_selection.py` (19 tests) — with-deps
  default (chroma under user data dir, valid config), without-deps default (memory, probe
  failure fails closed), explicit overrides (TOML, env var, constructor, from_dict,
  profile path), persist-dir overrides, and an importorskip-gated chroma persistence
  round-trip proving documents survive store recreation.
- Verified: `Tests/RAG/ Tests/RAG_Search/ Tests/UI/test_settings_library_rag_defaults.py
  Tests/test_smoke.py` → 329 passed / 19 skipped (baseline 310 passed / 20 skipped, delta
  = 19 new tests + smoke skip→pass). Runtime check: `create_rag_service("hybrid_basic")`
  now yields `ChromaVectorStore` (persistent) by default and `InMemoryVectorStore` with
  `RAG_VECTOR_STORE=memory`.
- Files: `tldw_chatbook/RAG_Search/simplified/config.py`,
  `Tests/RAG/simplified/test_vector_store_selection.py` (new), `Tests/test_smoke.py`.
