---
id: TASK-299
title: >-
  Unify hybrid retrieval leg key namespaces so RRF fusion can merge overlapping
  documents
status: To Do
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
- [ ] #1 A document returned by both the FTS leg and the vector leg is fused into one result with blended RRF contributions from both legs
- [ ] #2 Fusion unit tests cover cross-leg overlap with the real key shapes emitted by search_semantic and the FTS5 leg functions (not hand-built keys)
- [ ] #3 Semantic-leg citations survive on the fused result when the FTS item is primary (existing 681 test extended to the real-overlap path)
- [ ] #4 No change to single-leg ranking behavior (alpha semantics and RRF math unchanged)
<!-- AC:END -->
