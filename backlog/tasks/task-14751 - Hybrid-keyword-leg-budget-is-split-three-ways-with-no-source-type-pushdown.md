---
id: TASK-14751
title: Hybrid keyword-leg budget is split three ways with no source-type pushdown
status: To Do
assignee: []
created_date: '2026-08-09 21:21'
labels:
  - rag
  - retrieval
  - p2
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 eval harness cluster's final review (TASK-3994/3996). TASK-3996 fixed the keyword leg's media-only blindness by round-robining three sub-legs (media, notes, conversations) into one FIXED top_k FTS budget, while the Library's hybrid arm post-filters the fused rows by the user's selected source types AFTER fusion. Nothing tells the leg which types the user asked for, so it spends up to two thirds of its budget retrieving rows that are then discarded downstream.

Measured by the reviewer's probe (12 matching documents of each type seeded, leg asked for 20): the leg returned {media 7, note 7, conversation 6}. Media went from 20 rows before TASK-3996 to 7, and with Media-only selected the other 13 rows are thrown away rather than backfilled with media. The worst case is hybrid + Media-only + a thin or empty vector index - the ROUTE_NOTE_SEMANTIC_LEG_EMPTY "keyword-only results" case - where the user sees roughly one third of the media results dev returns for the same query.

This is invisible to both existing guards: the P1 eval gate selects all three source types, so the round robin is exactly what it wants and the metric never moves; and the Library unit tests drive canned fakes rather than a real mixed corpus, so no test observes the composition of a real leg's output. Note the trade-off being restored, not reverted: interleaving exists because FTS5 scores from different tables are not comparable and concatenation lets one well-stocked source consume every slot. The fix is to stop the leg budgeting for types the caller will discard, not to abandon rank-fair interleaving among the types it is actually asked for.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The caller's selected source types reach the keyword leg (pushdown), or equivalently the leg's top_k budget is allocated only across the selected types, so no budget is spent on rows that fusion's caller will discard.
- [ ] #2 A media-only hybrid search against an empty or thin vector index returns as many media rows as the pre-TASK-3996 leg did for the same query and top_k.
- [ ] #3 A test pins the leg's composition against a REAL mixed corpus (media, notes and conversations seeded in real DBs, as Tests/RAG_Search/test_keyword_leg_chacha.py does), not canned fakes - it must red if the budget silently reverts to a fixed three-way split under a single-type selection.
- [ ] #4 Rank-fair interleaving is preserved among the types that ARE selected (a multi-type selection must not regress to concatenation, where one well-stocked source consumes every slot).
<!-- AC:END -->
