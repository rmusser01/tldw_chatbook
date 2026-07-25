---
id: TASK-482
title: >-
  Validate Chroma persist_directory uniformly across vector_store and
  collection_indexes
status: Done
assignee:
  - '@claude'
created_date: '2026-07-22 00:45'
updated_date: '2026-07-25 15:24'
labels:
  - rag
  - security
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from PR #771 (RAG SP1) review: Qodo flagged that `collection_indexes._client()` passes the config-sourced `persist_directory` straight into Chroma's `PersistentClient` without going through `Utils/path_validation.py`. This is a **pre-existing, store-wide pattern** — `ChromaVectorStore` (`vector_store.py:199/273`) already does the same — so SP1 deliberately mirrored it (validating only the new module would risk a normalized-vs-raw path-string divergence from the store → the `SharedSystemClient` per-persist_directory client-cache collision that SP1's migration explicitly avoids).

Harden it **uniformly**: validate/normalize the Chroma persist_directory once, at a shared point, so `vector_store.py` and `collection_indexes.py` always receive the identical validated path. Do not introduce a normalization difference between the two Chroma client construction sites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Config-sourced Chroma `persist_directory` is validated via `path_validation.py` before use.
- [x] #2 `vector_store.py` and `collection_indexes.py` construct their `PersistentClient` with the SAME validated path string (no divergence → no `SharedSystemClient` Settings/path collision).
- [x] #3 Existing RAG tests still pass; a test covers that both sites resolve to the identical validated path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add validate_chroma_persist_directory(persist_directory) -> Path to RAG_Search/simplified/config.py, next to default_chroma_persist_directory(): expanduser() then route through Utils/path_validation.py's validate_path_simple() (no base-directory confinement, since RAG_PERSIST_DIR/config persist_directory may legitimately point anywhere -- validate_path would also false-positive on the common default ~/.local/share/tldw_cli/... dotted-ancestor path), re-raising a clear ValueError with context on rejection.\n2. Route ChromaVectorStore.__init__ (vector_store.py) through the shared helper when setting self.persist_directory, so the already-validated Path is reused by the client property (no double validation).\n3. Route collection_indexes._client() through the same shared helper before constructing PersistentClient.\n4. TDD: add Tests/RAG/simplified/test_chroma_persist_directory.py covering (a) identical validated path string at both real construction seams for the same input, (b) both sites reject an invalid/dangerous path identically (same ValueError), (c) legit absolute and ~-containing paths normalize stably (idempotent).\n5. Run both module test files + full Tests/RAG/ suite; update task-482 with Implementation Notes and mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added validate_chroma_persist_directory() to RAG_Search/simplified/config.py (next to default_chroma_persist_directory(), which already produces the value): expanduser() then Utils/path_validation.py's validate_path_simple() (chosen over validate_path -- no natural base directory to confine an arbitrary configured Chroma dir to, and validate_path would false-positive on the common default persist dir under a dotted ancestor, ~/.local/share/tldw_cli/...); wraps rejection in a clear ValueError with the original path for context. Both PersistentClient construction sites now route through this single helper: ChromaVectorStore.__init__ (vector_store.py) stores the validated Path directly on self.persist_directory (reused unchanged by the client property), and collection_indexes._client() calls it immediately before constructing the client. Proved the divergence this closes is real, not hypothetical: with only one site updated, a messy-but-legal input (double slash + trailing slash) normalized to two DIFFERENT path strings across the two sites (one Path-wrapped, one raw) -- exactly the SharedSystemClient collision trap described in the task. New tests in Tests/RAG/simplified/test_chroma_persist_directory.py mock chromadb.PersistentClient and call the real ChromaVectorStore.client / collection_indexes._client seams (not copies of the logic) to lock in: (1) identical validated path string at both sites for the same input, (2) identical ValueError rejection at both sites for a null-byte path, (3) stable/idempotent expanduser+absolute normalization. One existing test (test_vector_store_errors.py::test_persist_directory_permissions) needed its except clause widened to include ValueError: the new upfront validation's internal path.exists() probe can itself hit PermissionError while stat'ing a path under a directory it can't traverse, which path_validation.validate_path_simple wraps as ValueError -- documented inline in the test. Full Tests/RAG/ suite: 540 passed, 8 skipped (baseline 537 passed/8 skipped + 3 new tests), no regressions. Files: tldw_chatbook/RAG_Search/simplified/config.py, tldw_chatbook/RAG_Search/simplified/vector_store.py, tldw_chatbook/RAG_Search/simplified/collection_indexes.py, Tests/RAG/simplified/test_chroma_persist_directory.py (new), Tests/RAG/simplified/test_vector_store_errors.py.

Review follow-up: reviewer flagged that two persist_directory PRODUCERS skipped .expanduser(), so a literal '~' could reach the client sites unexpanded while validate_chroma_persist_directory (added above) expected an already-expanded value -- active_config.py's _apply_env_overrides (RAG_PERSIST_DIR env override layer, applied even on top of a profile's stored config) did config.vector_store.persist_directory = Path(persist), and RAGConfig.from_dict (saved/legacy profile JSON) did Path(vector_store_data['persist_directory']), neither expanding ~. Both now route through validate_chroma_persist_directory instead of a bare Path(...), so all four persist_directory touch points (2 producers + 2 consumers) share one normalization function; docstring updated to explain why production and consumption must agree. TDD: 2 new RED tests (test_env_var_persist_directory_expands_tilde_like_client_sites, test_from_dict_persist_directory_expands_tilde_like_client_sites) failed against the old code with PosixPath('~/x') != the expanded path, then passed after the fix. Verified ValueError from either producer is survivable: config_profiles.py's profile-loading loop already wraps ProfileConfig.from_dict (which calls RAGConfig.from_dict) in a broad except Exception that logs and skips the one bad file; resolve_active_rag_config()'s callers (create_rag_service et al.) are already broadly try/excepted at the RAG-service-construction layer, same as the original two consumer sites. Full Tests/RAG/ suite: 542 passed, 8 skipped (was 540/8 before this follow-up; +2 = the new tests, no regressions). Additional files touched: tldw_chatbook/RAG_Search/simplified/active_config.py.
<!-- SECTION:NOTES:END -->
