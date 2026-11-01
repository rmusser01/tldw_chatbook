---
id: TASK-391
title: >-
  Unify hybrid retrieval leg key namespaces so RRF fusion can merge overlapping
  documents
status: Done
assignee: []
created_date: '2026-07-19 04:23'
labels:
  - rag
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The RRF fusion shipped in task-256 (PR 681) blends FTS and vector legs by document identity, but the legs emit different result-key namespaces: search_semantic produces source='unknown' plus chunk-level ids while the FTS5 legs key by source document, so the same underlying document is never recognized as present in both legs. Consequences: cross-leg RRF score blending effectively never fires (each doc scores from one leg only), and the overlap citation-merge path added in the 681 review round is unreachable in practice. Flagged as a follow-up in PR 681's review discussion (comment 3604960547 reply) and the task-256 Implementation Notes. Unify the identity scheme (e.g. carry source_type plus source_id through search_semantic's result metadata into the leg key, falling back to chunk id only when no doc identity exists) so fusion sees true overlap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A document returned by both the FTS leg and the vector leg is fused into one result with blended RRF contributions from both legs
- [x] #2 Fusion unit tests cover cross-leg overlap with the real key shapes emitted by search_semantic and the FTS5 leg functions (not hand-built keys)
- [x] #3 Semantic-leg citations survive on the fused result when the FTS item is primary (existing 681 test extended to the real-overlap path)
- [x] #4 No change to single-leg ranking behavior (alpha semantics and RRF math unchanged)
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Fixed by TASK-3994/TASK-3996: _fusion_doc_key (RAG_Search/simplified/rag_service.py:673-694) matches the hybrid legs on document identity (source_type, source_id-or-doc_id), explicitly because 'the FTS leg ranks documents and the vector leg ranks chunks, so an id match was impossible'; the keyword leg stamps source_id so both legs share one namespace.
