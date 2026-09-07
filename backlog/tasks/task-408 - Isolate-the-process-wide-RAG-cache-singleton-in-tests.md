---
id: TASK-408
title: Isolate the process-wide RAG cache singleton in tests
status: To Do
assignee: []
created_date: '2026-07-21 09:48'
labels:
  - tests
  - rag
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
simple_cache.get_rag_cache is a first-caller-wins process singleton; cross-file test pollution twice disrupted the RAG scope program (Task 3 empirically; Task 6's combined Tests/Library+Tests/RAG run fails test_cache_hit — reproduced on clean base). PR 677's review already root-caused one instance (monkeypatch isolation). Add a shared autouse fixture (or reset seam) so suites can run combined in any order; remove the per-file workarounds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Combined Tests/Library + Tests/RAG run passes in either order with zero cache-pollution failures
- [ ] #2 Per-file clear() workarounds replaced by the shared isolation seam
<!-- AC:END -->
