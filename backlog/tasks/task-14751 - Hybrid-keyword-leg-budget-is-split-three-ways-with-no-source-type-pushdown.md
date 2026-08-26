---
id: TASK-14751
title: Hybrid keyword-leg budget is split three ways with no source-type pushdown
status: Done
assignee: []
created_date: '2026-08-09 21:21'
updated_date: '2026-08-10 22:27'
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
- [x] #1 The caller's selected source types reach the keyword leg (pushdown), or equivalently the leg's top_k budget is allocated only across the selected types, so no budget is spent on rows that fusion's caller will discard.
- [x] #2 A media-only hybrid search against an empty or thin vector index returns as many media rows as the pre-TASK-3996 leg did for the same query and top_k.
- [x] #3 A test pins the leg's composition against a REAL mixed corpus (media, notes and conversations seeded in real DBs, as Tests/RAG_Search/test_keyword_leg_chacha.py does), not canned fakes - it must red if the budget silently reverts to a fixed three-way split under a single-type selection.
- [x] #4 Rank-fair interleaving is preserved among the types that ARE selected (a multi-type selection must not regress to concatenation, where one well-stocked source consumes every slot).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. READ: confirm no positional callers of _keyword_search past top_k; map sub-leg budgeting (each sub-leg already gets full top_k; the three-way split happens at the interleave truncation).
2. RED tests in Tests/RAG_Search/test_keyword_leg_pushdown.py over REAL DBs (media + chacha writer APIs): media-only full budget (direct leg AND through hybrid over an empty vector index), unselected sub-legs never queried (spies), multi-type rank-fair interleaving, None == today, empty selection == [], Library vocabulary translation, cache-key separation.
3. Implement keyword-only kwarg keyword_source_types on search/_hybrid_search/_keyword_search/_chacha_keyword_sublegs/_chacha_fts_rows (None => all three; unknown values ignored with a debug log; empty => no sub-legs, no queries, []). Include the selection in the search cache key. Library _search_hybrid translates its plural scope through _ENGINE_KEYWORD_SOURCE_TYPES.
4. Mutation: drop the pushdown (ignore the kwarg) -> media-only + never-queried tests red, rest green; Edit-restore.
5. Battery + informational gated RAG_EVAL run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pushes the caller's source-type selection INTO the engine's keyword leg instead of letting fusion's caller discard rows the leg already paid for.

Approach: a keyword-only `keyword_source_types: Collection[str] | None` on `RAGService.search` / `_hybrid_search` / `_keyword_search` (plus `source_types` on `_chacha_keyword_sublegs`/`_chacha_fts_rows`). `None` means all three sub-legs, so every pre-existing caller is byte-for-byte unchanged; unknown values are dropped with a debug log (fail open to fewer sub-legs, never a crash); an empty collection means no sub-legs at all and returns [] without touching a database (hybrid then degrades to semantic through the existing disclosed route). Only the selected sub-legs RUN -- a media-only leg does not even open the ChaChaNotes DB -- so the whole top_k budget goes to types whose rows survive. Interleaving is untouched: it is now applied among exactly the selected types, which is the trade-off being restored rather than reverted (FTS5 scores from different tables are not comparable).

Library `_search_hybrid` translates its plural scope through a pure map (`_ENGINE_KEYWORD_SOURCE_TYPES`: media->media, notes->note, conversations->conversation; prompts has no engine seam). The engine's vocabulary is singular, and a plural handed down would be dropped as unknown -- i.e. an EMPTY keyword leg -- so a test pins both the translation and the map's domain against `_FTS_SERVABLE_SOURCE_TYPES`.

Two things the work turned up beyond the plan:
* The selection had to enter the search CACHE key (`SimpleRAGCache._make_key`/`get_async`/`put_async`, `kst:` part, omitted when None so old keys are unchanged). Without it a media-only and a notes-only search of the same query/top_k share an entry and the second is served the first's rows -- reinstating the defect for every query after the first. The sync `get`/`put` twins deliberately stay as they are: no caller can hand them a selection, so the worst a mixed workload yields is a miss.
* `EnhancedRAGServiceV2` -- the class actually instantiated at runtime -- overrides `search()` with an explicit signature, so it did NOT inherit the new kwarg. The Library's pushdown crashed the P1 eval harness with a TypeError while all 12 unit tests stayed green, because every double mirrors `RAGService.search` rather than the override. Fixed and pinned by a test that drives the real subclass.

Evidence: Tests/RAG_Search/test_keyword_leg_pushdown.py (22 tests, real media + ChaChaNotes DBs via the writer APIs -- canned fakes are the blindness that let this defect live). The AC#2 pin runs the HYBRID path against an unindexed vector store (the 'semantic leg empty -- keyword-only results' route the defect was worst on): 25 matching media, top_k=20 -> 20 media rows = min(N, top_k), the pre-TASK-3996 full budget; the same call without a selection returns 7 and is asserted in the same test so the pin cannot be satisfied by a corpus too small to show it. `test_none_means_all_three_sub_legs_unchanged` reproduces the reviewer's measured probe exactly (12/12/12, top_k=20 -> media 7, note 7, conversation 6). Mutation (ignore the kwarg): 9 failed / 45 passed across the pushdown + chacha + Library-mode files -- the nine pushdown pins red (including the runtime-class one), the backward-compat, vocabulary, cache-key-unit and semantic-guard pins green; a second mutation (drop the forward in the V2 override) reds only the runtime-class pin. Battery 421 passed / 6 skipped across the new file + test_keyword_leg_chacha + test_keyword_leg_db_resolution + test_hybrid_doc_fusion + the two Library RAG suites + Tests/DB/test_private_sqlite_inventory + Tests/RAG/simplified. Gated harness (RAG_EVAL=1 Tests/RAG_Eval): 142 passed, every metric +0.000 in all three modes -- expected and gate-invisible BY CONSTRUCTION, since the harness selects all three source types and therefore exercises the None path.

Files: tldw_chatbook/RAG_Search/simplified/rag_service.py, simple_cache.py, enhanced_rag_service_v2.py; tldw_chatbook/Library/library_local_rag_search_service.py; Tests/RAG_Search/test_keyword_leg_pushdown.py (new); Tests/Library/test_library_rag_mode_resolution.py (double mirrors the real signature).

Review round (spec pass, 2026-08-10): three findings folded in.

1. The cache-key coverage pinned only the MISS direction. The HIT direction -- the same selection in any iteration order must produce ONE key -- was unpinned, so deleting the `sorted()` in `_make_key` would silently turn every mixed-selection workload into a permanent cache miss with zero red tests. Added direct `_make_key` pins: same selection as list / reversed tuple / frozenset / set collapses to one key; `None` reproduces the byte-identical legacy five-argument key; `set()` stays distinct from `None`; different selections stay distinct. The list and tuple cases are deliberate reverses of each other -- a set's iteration order is fixed within a process, so it can agree with at most one of them, and dropping either would leave a per-hash-seed coin flip. Verified: `sorted(...)` -> `list(...)` reds exactly one of the two.
2. The semantic + keyword_source_types ValueError guard was asserted only through the Library suite's double -- the same base-class-vs-runtime blindness recorded in this task's own lessons entry. Added a real-engine test; removing the guard now reds it.
3. The unknown-source-type debug log is now asserted (loguru sink at DEBUG, the test_keyword_leg_chacha idiom -- capsys never sees loguru), including that it NAMES the dropped types. Removing the log reds it. Failing open is only defensible if it leaves a trace.
<!-- SECTION:NOTES:END -->
