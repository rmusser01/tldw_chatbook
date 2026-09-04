---
id: TASK-31260
title: Library wiring cluster-membership AST re-census guard
status: To Do
assignee: []
created_date: '2026-09-04 05:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recipe section 16 lesson 5 flagged the Library wiring-cluster method-name tuples (`_EXPORT_CLUSTER_METHOD_NAMES`, `_COLLECTIONS_CLUSTER_METHOD_NAMES`, and the equivalents now hand-kept by the conversations and search+RAG wiring tests) as frozen, hand-written snapshots that nothing re-verifies against the live LibraryScreen source. A future same-named method landing on the screen -- genuinely subsystem-owned or a same-named coincidence -- is invisible to every wiring and architecture test that exists today; it is not flagged as needing a cluster-membership decision. This closes the deferred gap recipe section 16 lesson 5 named as a wave-3-or-later candidate, now that a third and fourth hand-kept tuple (conversations, search+RAG) exist alongside the original two (export, collections).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A guard test fails when a search/rag/export/collections/conversations-named method is added to or removed from LibraryScreen without a matching update to its corresponding hand-kept wiring-cluster tuple
- [ ] #2 The guard test passes against the current, unmodified tree, proving today's four hand-kept tuples (export, collections, conversations, search+RAG) are accurate at filing time
<!-- AC:END -->
