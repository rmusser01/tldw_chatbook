---
id: TASK-14752
title: Hybrid coverage copy can mis-describe keyword-sourced evidence
status: Done
assignee: []
created_date: '2026-08-09 21:21'
updated_date: '2026-08-11 15:35'
labels:
  - rag
  - library
  - ux-copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 eval harness cluster's final review (TASK-3994/3996). The Library's scope-coverage diagnostic renders "Semantic search found nothing from: <types>." (`_semantic_scope_coverage` in `Library/library_local_rag_search_service.py`, wording built in `library_rag_state.py`). Before TASK-3996 the engine's keyword leg served media only, so under a hybrid profile a type with no semantic hits also had no evidence at all and the sentence read correctly.

TASK-3996 gave that leg notes and conversation sub-legs. The sentence can now be literally true of a source type whose on-screen evidence came entirely from the keyword leg - the user reads "found nothing from Notes" while looking at note rows in the results list. The claim is not false (it is scoped to the semantic leg), but it invites the reader to conclude the opposite of what the screen shows, and "no semantic hits" is not information a user can act on when the rows are there.

This was noticed while implementing TASK-3996, deliberately left unbundled from the retrieval fix as a UI-copy follow-up, and flagged in-code at the `_hybrid`/coverage seam - but it was never given a task number. This task is that number.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The coverage note distinguishes "no semantic hits for this type" from "no evidence at all for this type" when the keyword leg supplied rows for that type, so a user never reads that a type produced nothing while its rows are on screen.
- [x] #2 A test pins the mixed case: under a hybrid profile with keyword-sourced rows present and no semantic hits for a selected type, the rendered copy is the "keyword-only" wording, not the bare "Semantic search found nothing" sentence.
- [x] #3 The semantic-only and plain profiles' existing copy is unchanged (the pure-semantic case still reads exactly as it does today).
- [x] #4 The in-code follow-up note at the coverage seam is replaced by a reference to this task's resolution rather than left dangling.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Rides inside TASK-15020's B1 (scope-aware hybrid): this task's coverage-copy fix touches the same disclosure seams B1 rewrites (Library's _semantic_scope_coverage / library_rag_state.py wording, and the ROUTE_NOTE_HYBRID_SCOPED family) -- landing it separately would collide with B1's edits to those seams. See Docs/superpowers/specs/2026-08-11-rag-p2ab-instrument-and-deferred-constraints-design.md (B1) and Docs/superpowers/plans/2026-08-11-rag-p2ab-instrument-and-deferred-constraints.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed inside TASK-15020's B1b (the Library routing task), as the plan required.

Approach: made the coverage diagnostic leg-aware rather than rewording the sentence in place. `_semantic_scope_coverage` (Library/library_local_rag_search_service.py) now partitions the requested, semantically-coverable source types three ways instead of two: `covered` (a row of that type carried a vector-leg contribution), `keyword_only` (rows are present but every one is FTS-only), `uncovered` (no rows at all). The per-row judgement lives in a new `_row_is_keyword_only`, which `_rows_are_keyword_only` now delegates to, so the whole-set 'semantic leg empty' claim and the per-type claim cannot drift into two readings of the same fusion block. A row with no fusion block is deliberately NOT keyword-only: that default is what keeps every non-hybrid search byte-identical and stops the fix from inventing the mirror-image false claim out of absent provenance.

`keyword_only` is OMITTED from the payload when empty, so the semantic and plain profiles' diagnostics -- and therefore their copy -- are unchanged (AC#3). `library_rag_coverage_note` (Library/library_rag_state.py) renders the new list as its own sentence, 'Keyword matches only from: <types>.', after the 'Semantic search found nothing from: ...' sentence and before the route notes; both label lists now share one escaped display-vocabulary helper (`_coverage_labels`).

Trade-off accepted: under hybrid, a type with NO rows still reads 'Semantic search found nothing from: X.' even though the keyword leg also found nothing for it. Accurate but incomplete; widening it was outside this AC.

AC#4: the dangling in-code follow-up note in `_retrieval_payload`'s docstring ('a hybrid-aware wording variant is a UI-copy follow-up ... deliberately not bundled') now describes this task's resolution instead.

Tests (RED-first): 5 copy tests in Tests/Library/test_library_rag_state.py (keyword-only sentence; keyword-only + uncovered as two sentences; display-label vocabulary + unknown-type fallback; absent key => byte-identical old copy; zero-row suppression) and 2 diagnostics tests in Tests/Library/test_library_rag_mode_resolution.py (mixed hybrid rows partition three ways; a fusion-block-less row is never called keyword-only). Mutation-checked in both directions: reverting the wording reds 3 copy tests; forcing `_row_is_keyword_only` to False reds the diagnostics test AND the pre-existing 'index empty not claimed' test, proving the shared delegation is live.

Docs: Docs/User_Guide/library/search-and-rag.md's coverage-note paragraph gains the keyword-only case (its live-check stamp is deliberately left to the arc's live-check task).

Modified: tldw_chatbook/Library/library_local_rag_search_service.py, tldw_chatbook/Library/library_rag_state.py, Tests/Library/test_library_rag_state.py, Tests/Library/test_library_rag_mode_resolution.py, Docs/User_Guide/library/search-and-rag.md.
<!-- SECTION:NOTES:END -->
