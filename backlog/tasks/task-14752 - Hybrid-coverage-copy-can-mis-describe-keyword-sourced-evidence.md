---
id: TASK-14752
title: Hybrid coverage copy can mis-describe keyword-sourced evidence
status: To Do
assignee: []
created_date: '2026-08-09 21:21'
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
- [ ] #1 The coverage note distinguishes "no semantic hits for this type" from "no evidence at all for this type" when the keyword leg supplied rows for that type, so a user never reads that a type produced nothing while its rows are on screen.
- [ ] #2 A test pins the mixed case: under a hybrid profile with keyword-sourced rows present and no semantic hits for a selected type, the rendered copy is the "keyword-only" wording, not the bare "Semantic search found nothing" sentence.
- [ ] #3 The semantic-only and plain profiles' existing copy is unchanged (the pure-semantic case still reads exactly as it does today).
- [ ] #4 The in-code follow-up note at the coverage seam is replaced by a reference to this task's resolution rather than left dangling.
<!-- AC:END -->
