---
id: TASK-15672
title: >-
  Lazy embeddings_rag dependency-check guard is order-dependent:
  DEPENDENCIES_AVAILABLE left False by an earlier suite
status: To Do
assignee: []
created_date: '2026-08-11 20:09'
labels:
  - bug
  - rag
  - tests
  - pre-existing
  - order-dependent
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/RAG/test_lazy_embeddings_rag_dependency_check.py::test_create_rag_service_succeeds_for_backfill_without_a_prior_eager_check passes in isolation but fails after Tests/RAG_Search has run: create_rag_service() succeeds, yet DEPENDENCIES_AVAILABLE['embeddings_rag'] is still False, so the assertion that the lazy check actually ran fails. The module-level DEPENDENCIES_AVAILABLE dict is shared process state that an earlier suite leaves in a stubbed/false condition. This is the regression guard for TASK-657 (Done), so while it is order-dependent that fix is unguarded in any full-suite run. Pre-existing: reproduces identically on merge-base ced98b9a4, and was surfaced by the TASK-15020 closing battery, not caused by it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes both in isolation and after Tests/RAG_Search in the same process
- [ ] #2 Whatever mutates DEPENDENCIES_AVAILABLE restores it (or the test establishes its own state) so the guard does not depend on collection order
- [ ] #3 The guard still fails if the lazy embeddings_rag dependency check stops running, i.e. it is repaired rather than relaxed
<!-- AC:END -->
