---
id: TASK-297
title: >-
  Align hybrid retrieval fusion with the tldw_server design (RRF plus alpha
  weighting)
status: To Do
assignee: []
created_date: '2026-07-12 14:12'
labels:
  - rag
dependencies:
  - TASK-288
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_server fuses hybrid results with Reciprocal Rank Fusion (k=60) followed by an alpha-weighted blend of the FTS and vector RRF scores, with alpha=0.7 weighting the vector side (tldw_server2 database_retrievers.py:2044-2092). The chatbook hybrid pipeline is an ad-hoc weighted merge whose vector leg has always been empty in practice. Once indexing makes the vector leg real (task-247), align the fusion math and defaults with the server so hybrid result quality and behavior match the reference design. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Hybrid mode fuses FTS and vector rankings via RRF with an alpha-weighted blend matching server defaults (k=60, alpha=0.7 vector-weighted)
- [ ] #2 Fusion is covered by unit tests using known input rankings
- [ ] #3 Alpha is exposed in config with a server-consistent default and documented semantics (0 = FTS only, 1 = vector only)
<!-- AC:END -->
