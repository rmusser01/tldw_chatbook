---
id: TASK-31260
title: Library wiring cluster-membership AST re-census guard
status: Done
assignee:
  - '@zcode'
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
- [x] #1 A guard test fails when a search/rag/export/collections/conversations-named method is added to or removed from LibraryScreen without a matching update to its corresponding hand-kept wiring-cluster tuple
- [x] #2 The guard test passes against the current, unmodified tree, proving today's four hand-kept tuples (export, collections, conversations, search+RAG) are accurate at filing time
<!-- AC:END -->

## Implementation Notes

``Tests/Architecture/test_library_cluster_membership_census.py`` re-derives all FIVE hand-kept clusters (export, collections, conversations reader, conversations browse, search+RAG -- the task's four, with conversations holding two tuples) against the live ``LibraryScreen`` source and asserts, per cluster:

    screen census(pattern) == ((MOVED - PRUNED) - NO_DELEGATOR) | STAYED

MOVED/PRUNED load LIVE from each owning wiring test's tuples, so a real membership change recorded there passes without touching the census. Only two small frozen lists live in the census, each with in-file guidance: ``_NO_DELEGATOR`` (11 moved-unpruned names with no screen delegator -- e.g. browse's ten state-field accessors, reader's ``find_in_library_conversation`` action entry point) and ``_STAYED`` (pattern-matching screen methods outside the tuples -- the export @work stayers, reader-overlap names browse's broader pattern sweeps, etc.). One cross-cluster overlap was surfaced and recorded at freeze time: ``find_in_library_conversation`` is reader-owned yet swept by browse's ``library_conversation`` pattern.

TDD/mutation evidence (AC#1): renaming ``handle_library_search_clear`` on the screen made the census fail naming BOTH the unaccounted addition and the still-claimed removal, pointing at the owning wiring module; mutation reverted, census green on the unmodified tree (AC#2 -- all five tuples verified accurate at filing). Ruff clean after format.

ADR required: no
ADR path: N/A
Reason: Test-only census guard over existing wiring data; no production change.

Added: ``Tests/Architecture/test_library_cluster_membership_census.py``.
