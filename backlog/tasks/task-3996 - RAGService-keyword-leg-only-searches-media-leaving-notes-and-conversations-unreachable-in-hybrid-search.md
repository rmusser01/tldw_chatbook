---
id: TASK-3996
title: >-
  RAGService keyword leg only searches media, leaving notes and conversations
  unreachable in hybrid search
status: Done
assignee: []
created_date: '2026-08-09 05:17'
updated_date: '2026-08-09 20:40'
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
- [x] #1 The keyword leg of hybrid search can return notes and conversations, not only media, when the query matches their content.
- [x] #2 A regression test with a notes-only or conversations-only relevant document confirms it is reachable through hybrid search FTS leg.
- [x] #3 The P1 eval harness baselines are re-stamped in the same PR, with before and after numbers included in the PR description.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-09-rag-port-hybrid-fusion-fixes.md (Task 5) and Docs/superpowers/specs/2026-08-09-rag-port-hybrid-fusion-fixes-design.md for the read-only notes/conversations sub-leg design.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The engine's FTS leg is now three sub-legs instead of one media query.
`_keyword_search` gathers the media sub-leg (`_media_keyword_subleg`, the old
body unchanged) and two new ChaChaNotes sub-legs in parallel, then merges them
with fusion.py's `interleave_rankings` (round robin by rank position, keyed on
`_fusion_doc_key`) and trims to `top_k`. FTS5 scores from different tables are
not comparable, so rank position is the only honest cross-source signal, and
concatenation would let media consume every slot.

**Read-only, no ORM.** `_connect_chacha_readonly` opens
`sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)`: structurally unable
to write, and never `CharactersRAGDB` (whose constructor does schema and
client-registration work on the user's main DB). `_chacha_notes_fts` and
`_chacha_conversations_fts` mirror `search_notes` /
`search_conversations_by_content` including every soft-delete filter
(`notes.deleted = 0`; `messages.deleted = 0` AND `conversations.deleted = 0`).
Conversations are per-conversation rows (`GROUP BY c.id`, `MIN(rank)`), so the
keyword `source_id` is the conversation id the vector leg also stamps.

**Vocabulary equality is the whole point.** Rows stamp `source_type`
(`note`/`conversation`, exact singular) and `source_id` (bare row id) —
what `_fusion_doc_key` compares — so a keyword row and its vector twin fuse
into one row. `test_cross_leg_merge_per_source_type` builds the vector side
from the real `ingestion_indexing` document builders, so drift between the
two vocabularies fails a test rather than silently un-merging. Media rows now
stamp `source_id` too (previously only the prefixed id was available
downstream).

**Path handling** mirrors P0's `media_db_path` treatment exactly:
`config.search.chachanotes_db_path` (new) → `get_chachanotes_db_path()`,
through `validate_path_simple` + `lexical_path`, plus an explicit symlink
refusal (this leg opens SQLite directly, so it cannot inherit
`connect_private_sqlite`'s no-follow guarantee) and no create-on-miss. A
missing or unopenable DB costs only its sub-legs, with one warning; media is
unaffected, and the leg is empty only when every sub-leg is.

**Testing.** `Tests/RAG_Search/test_keyword_leg_chacha.py` (6 tests, 7 cases),
RED first, fixtures written through the real writer APIs. Mutation-checked:
plural vocabulary, each of the three soft-delete filters, concatenation
instead of interleaving, and `mode=rw` each red exactly the intended test.
The soft-delete tests were VACUOUS at first for notes — the trigger already
evicts a soft-deleted row from the external-content index — so the test now
rebuilds `notes_fts`/`messages_fts` (the state a maintenance rebuild leaves,
which this repo issues for two other tables) and the predicates become
observable.

**Harness.** `Tests/RAG_Eval/harness/ingest.py` now injects
`chachanotes_db_path` alongside `media_db_path`; without it the harness's FTS
leg resolved the real user path (absent under test isolation) and would have
kept measuring media only.

AC #3 (baseline re-stamp with before/after numbers) is deliberately NOT
ticked: the arc's Task 6 owns the single deliberate re-stamp. Informational
gated run: gate PASSES; hybrid overall P 0.117 -> 0.105, F1 0.208 -> 0.190,
R/MRR/NDCG unchanged at 1.000, mean distinct docs 9.1 -> 10.0 (the leg now
contributes rows that survive fusion; P@k divides by min(k, len(retrieved))).
Full numbers in .superpowers/sdd/2026-08-09-rag-port-hybrid-fusion-fixes/
task-5-report.md.

Modified: `tldw_chatbook/RAG_Search/simplified/rag_service.py`,
`tldw_chatbook/RAG_Search/simplified/config.py`,
`Tests/RAG_Eval/harness/ingest.py`. Added:
`Tests/RAG_Search/test_keyword_leg_chacha.py`.

**Plan Task 6 closure (re-stamp + live).** AC #3 ticked. Like TASK-3995, this
fix's own contribution to the stamped numbers was ZERO on the gated metrics -
the informational run taken right after it moved nothing, because every golden
query whose FTS leg fires already had its target at vector rank 1, so reaching
notes and conversations added coverage the metrics could not see. That is a
property of the corpus, not evidence the fix is inert, and it is exactly why
plan Task 6 added a vector-blind fixture (see TASK-3994's AC #2 and TASK-4110).

Live verification (2026-08-09, tmux, scratch profile holding a copy of the real
Library DBs and vector index): Library > Search/RAG, RAG Answer mode, default
Hybrid Basic profile, **Media deselected** with Notes and Conversations in
scope - the case that used to fall back to semantic and now must not. The query
"worktree UAT database" returned a note row and a conversation row, both banded
"keyword match". Neither source type was reachable through the engine's keyword
leg before this change, and the vector index on that profile holds media chunks
only, so both rows can only have come from the new ChaChaNotes sub-legs.
<!-- SECTION:NOTES:END -->
