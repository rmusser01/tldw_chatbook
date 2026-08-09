---
id: TASK-3998
title: >-
  Eval-harness fingerprint must record the load-bearing stack
  (transformers/torch/chromadb), not sentence-transformers
status: In Progress
assignee: []
created_date: '2026-08-09 14:48'
updated_date: '2026-08-09 17:19'
labels:
  - rag
  - eval
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 final review (TASK-3894): the harness's real embedding path is Embeddings_Lib._HFEmbedder -> transformers.AutoModel + torch, with chromadb doing ANN retrieval; none of those three is fingerprinted, while sentence-transformers -- which is not on the load path -- is. This breaks in both directions: upgrading torch/transformers/chromadb shifts numerics with NO fingerprint change, producing a false REGRESSED hunt; upgrading sentence-transformers alone produces a pointless ENVIRONMENT_CHANGED re-stamp with no actual numeric change behind it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fingerprint includes transformers, torch, and chromadb versions
- [ ] #2 The sentence-transformers key's retention or removal is decided and documented
- [ ] #3 Baselines are re-stamped in the same commit with both old and new fingerprints shown
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-09-rag-port-hybrid-fusion-fixes.md (Task 2) and Docs/superpowers/specs/2026-08-09-rag-port-hybrid-fusion-fixes-design.md for the fingerprint-keys design (transformers/torch/chromadb compared, sentence-transformers informational).
<!-- SECTION:PLAN:END -->
