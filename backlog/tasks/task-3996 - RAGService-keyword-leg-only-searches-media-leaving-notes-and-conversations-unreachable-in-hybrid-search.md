---
id: TASK-3996
title: >-
  RAGService keyword leg only searches media, leaving notes and conversations
  unreachable in hybrid search
status: In Progress
assignee: []
created_date: '2026-08-09 05:17'
updated_date: '2026-08-09 17:19'
labels:
  - rag
  - retrieval
  - p2
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 eval harness (TASK-3894). RAGService._perform_fts5_search (rag_service.py, near L1340-1355) is hardcoded to FROM Media m JOIN media_fts ON m.id = media_fts.rowid, so the keyword leg of hybrid search can only ever return media documents. On the P1 fixture corpus, 28 of 48 documents are notes or conversations and are structurally unreachable by this leg regardless of query content, confirmed by source inspection. The four-seam keyword path (Library/library_fts_query.py) already searches media, notes, and conversations and does not share this limitation; this task is the engine-leg half of that same P2 scope note.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The keyword leg of hybrid search can return notes and conversations, not only media, when the query matches their content.
- [ ] #2 A regression test with a notes-only or conversations-only relevant document confirms it is reachable through hybrid search FTS leg.
- [ ] #3 The P1 eval harness baselines are re-stamped in the same PR, with before and after numbers included in the PR description.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-09-rag-port-hybrid-fusion-fixes.md (Task 5) and Docs/superpowers/specs/2026-08-09-rag-port-hybrid-fusion-fixes-design.md for the read-only notes/conversations sub-leg design.
<!-- SECTION:PLAN:END -->
