---
id: TASK-3995
title: >-
  RAGService keyword leg wraps every query in phrase quotes, blocking
  non-contiguous multi-token matches
status: Done
assignee: []
created_date: '2026-08-09 05:16'
updated_date: '2026-08-09 20:40'
labels:
  - rag
  - retrieval
  - p2
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 eval harness (TASK-3894). RAGService._escape_fts5_query (rag_service.py, near L1289-1301) wraps the entire keyword query in double quotes before running it against media_fts, which turns every multi-token search into an FTS5 phrase query requiring a contiguous token sequence -- strictly stronger than AND, not equivalent to it. Verified directly against a real corpus document and the exact SQL join the engine uses: the phrase form of a four-token query ("Obsidian-3 lathe spindle runout") matched 0 rows against a document that contains the relevant terms but not as one contiguous run, while an AND-of-terms form matched (a second multi-token query, "Calyx-77 torque limiter slipping", showed the same 0-rows phrase-query behavior directly against the live engine). The whole-query quoting is load-bearing safety, not an accident: the same probe's unquoted form, Obsidian-3 lathe spindle runout, raises OperationalError(no such column: 3) because FTS5 parses unquoted text as a column-filter expression. A fix must drop phrase semantics while keeping that safety, for example per-token quoting joined with AND, the approach Library/library_fts_query.py already uses for the four-seam path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Multi-token keyword queries match documents whose relevant tokens are present but not contiguous (AND-of-terms semantics or better), not only exact phrase matches.
- [x] #2 A query containing a token FTS5 would otherwise parse as a column filter (for example a bare hyphenated-numeric token) does not raise an FTS5 syntax or column error.
- [x] #3 Regression tests cover both the non-contiguous-match case and the injection-safety case against real corpus content.
- [x] #4 The P1 eval harness baselines are re-stamped in the same PR if hybrid or keyword numbers move, with before and after numbers included in the PR description.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-09-rag-port-hybrid-fusion-fixes.md (Task 3) and Docs/superpowers/specs/2026-08-09-rag-port-hybrid-fusion-fixes-design.md for the per-token FTS5 quoting design (AND-of-terms, safety preserved).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced whole-query phrase quoting in _escape_fts5_query (rag_service.py)
with per-token quoting joined by spaces (FTS5's implicit AND): each
whitespace-split token has its embedded double-quotes doubled and is
individually wrapped in quotes; a token with no alphanumeric character
(pure punctuation) is dropped since FTS5's default tokenizer can never
index it. If every token is dropped the function returns "" -- callers
must treat that as "no results".

Added a short-circuit at the top of _keyword_search: if
_escape_fts5_query(query) == "", return [] immediately, before resolving
the media DB path or acquiring a connection (no FTS5 call, no DB touch).

Safety property preserved: a bare hyphenated-numeric token like
"Obsidian-3" still raises OperationalError('no such column: 3') if passed
to FTS5 unquoted (verified with a real in-memory FTS5 table); quoting it
per-token keeps it safe exactly as whole-query quoting did.

New tests (Tests/RAG_Search/test_fts5_query_escaping.py) build a plain
sqlite3 stdlib in-memory FTS5 table (no app DBs) and exercise the real
_escape_fts5_query against it: non-contiguous multi-token match (RED
under the old phrase form, confirmed by asserting the old form still
returns 0 rows as a sanity check), hyphen-numeric safety, embedded-quote
safety, single-token behavior unchanged, and the all-punctuation
short-circuit (asserts _perform_fts5_search is never called).

P0's Tests/RAG_Search/test_keyword_leg_db_resolution.py (7 tests) verified
unmodified and green -- its seeds are single-token/contiguous, unaffected
by the AND-of-terms change.

Informational RAG_EVAL=1 gated run: gate held, 0 metric movement (all 60
gated metrics +0.000 vs baseline; "PASSED: No regression"). Expected and
explained: that harness's "plain" mode goes through the Library seam's own
build_fts_match_query grammar, not RAGService._escape_fts5_query, and its
"hybrid"/"semantic" modes don't yet route through the engine's keyword leg
either (fusion wiring is TASK-3994, not yet landed) -- so this leg's fix is
correctly invisible to that battery for now. Not re-stamping per the arc
plan; TASK-3995's AC #4 (re-stamp) is left unticked, to be closed in the
arc's final task in the same PR.

Files: tldw_chatbook/RAG_Search/simplified/rag_service.py
(_escape_fts5_query, _keyword_search); Tests/RAG_Search/test_fts5_query_escaping.py (new).

**Plan Task 6 closure (re-stamp).** AC #4 ticked. The arc's single deliberate
re-stamp landed in this PR. This fix's own contribution to it was measured
separately and was ZERO: the informational gated run taken immediately after
it moved all 60 gated metrics by +0.000, because the engine keyword leg's rows
could not yet survive fusion (TASK-3994, not yet landed at that point) and the
harness's plain mode uses the Library seam's own grammar. The movement in the
stamped baselines belongs to TASK-3994 and to the corpus addition; the
progression table in Tests/RAG_Eval/README.md attributes each step.

Live confirmation (2026-08-09, scratch profile over a copy of the real DBs):
the query "worktree UAT database" returned a note whose three tokens appear
non-adjacently. Under the previous whole-query phrase form that query matched
nothing at all.
<!-- SECTION:NOTES:END -->
