---
id: TASK-40440
title: Core SQLite owner privacy test cannot check the RAG indexing store since it moved to the store kernel
status: To Do
assignee: []
created_date: '2026-11-12 12:00'
labels:
  - testing
  - database
  - privacy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/DB/test_core_sqlite_owner_privacy.py` checks that each core SQLite owner opens its database through the literal private-SQLite seam, keeps its connection contract and never resolves the selected path. Its `rag_indexing` case cannot run any of that. Since TASK-33132 (`cbb39dd767`, `SQLiteStoreKernel`), `tldw_chatbook/DB/RAG_Indexing_DB.py` opens connections through `store_kernel` and no longer imports `connect_private_sqlite`, so the test's `monkeypatch.setattr(case.module, "connect_private_sqlite", ...)` raises `AttributeError` before the store is built.

Five cases are red on origin/dev `4b90eb860b` (2026-11-12), all with that `AttributeError`:
- `test_core_owner_uses_literal_seam_and_preserves_connection_contract[rag_indexing-first-create]`
- `test_core_owner_uses_literal_seam_and_preserves_connection_contract[rag_indexing-harden]`
- `test_core_owner_preserves_exact_memory_behavior[rag_indexing]`
- `test_core_owner_normalizes_relative_path_lexically[rag_indexing]`
- `test_domain_connection_exception_translation_is_preserved[rag_indexing-OperationalError]`

The test file's own comment on the case and TASK-39117's notes both record the red, but no task owns it, and no PR lane runs the file (only nightly-deep's whole-suite run reaches it), so nothing on a PR flags it. The privacy contract the file guards therefore goes unchecked for the RAG indexing store. Found while re-gating PR #3443 (TASK-33620.15.47), which touches this test file but not this case: the same five fail identically on that branch and on origin/dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rag_indexing cases observe the connection seam the RAG indexing store actually uses and pass on the current code
- [ ] #2 Each rag_indexing case fails when the store resolves its selected database path or bypasses the private-SQLite seam (shown by a temporary mutation recorded in the notes)
- [ ] #3 The other owner cases in the file keep their current results
<!-- AC:END -->
