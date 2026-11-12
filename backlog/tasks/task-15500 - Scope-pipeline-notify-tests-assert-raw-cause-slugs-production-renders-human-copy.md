---
id: TASK-15500
title: >-
  Scope-pipeline notify tests assert raw cause slugs; production renders human
  copy
status: Done
assignee: []
created_date: '2026-08-11 19:48'
labels:
  - bug
  - rag
  - tests
  - pre-existing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four tests in Tests/RAG/test_scope_pipeline_enforcement.py assert that a raw cause slug ("deleted-items", "workspace-read-failed") appears in the user-facing notify string, but production has rendered human copy ("the scoped items have been deleted") since dev commit 6f3b9b6d3 (a Chat-side change). The tests encode the old contract, so they are red on dev and the real notify copy is unguarded. Decide which side is the contract and make one of them follow the other.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tests/RAG/test_scope_pipeline_enforcement.py passes on a clean dev checkout
- [x] #2 The assertions check the copy the user actually sees, or the production seam is changed to emit what they check — with the choice recorded
- [x] #3 The user-facing notify copy for each scope-failure cause is pinned by a test, so a future copy change cannot silently pass
<!-- AC:END -->

## Closeout audit 2026-10-11

Board-prune audit (branch chore/board-prune-c, base a3c47c6c72): completed-but-unmarked. Evidence gathered 2026-10-11 against origin/dev:

AC dispositions:
- AC#1: met - The four raw-slug notify assertions no longer exist; every notify assertion now checks the rendered notice. In this audit's focused run all notify-related tests in the file passed (the file's 4 remaining local failures are Backup_Recovery bootstrap setup deaths, unrelated class).
- AC#2: met - Assertions now check the copy the user sees: tests assert scope_empty_notice(...) output (Chat/rag_scope.py:84), never a raw cause slug.
- AC#3: met - The user-facing notice is pinned via the scope_empty_notice/scope_cause_phrase seam used in assertions (e.g. test_no_workspace_overlap_short_circuits_empty_with_honest_notify).
- Commit 7fec8a1ab7 'test: align RAG scope notification evidence' (2026-08-14, after filing 2026-08-12) rewrote the assertions.
- Production seam tldw_chatbook/Chat/rag_scope.py:84 scope_empty_notice renders the mapped phrase; tests import and assert it.
- - Run caveat: this audit's local runs of the UI half died in fixture setup with Backup_Recovery bootstrap.RecoveryRequired('raw_source_selection_changed') - this machine's shared profile state, the repo's known local-profile flake class (see commit e1fee9bb1e), not a dev-tree assertion failure.
