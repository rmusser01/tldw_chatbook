---
id: TASK-3995
title: >-
  RAGService keyword leg wraps every query in phrase quotes, blocking
  non-contiguous multi-token matches
status: To Do
assignee: []
created_date: '2026-08-09 05:16'
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
- [ ] #1 Multi-token keyword queries match documents whose relevant tokens are present but not contiguous (AND-of-terms semantics or better), not only exact phrase matches.
- [ ] #2 A query containing a token FTS5 would otherwise parse as a column filter (for example a bare hyphenated-numeric token) does not raise an FTS5 syntax or column error.
- [ ] #3 Regression tests cover both the non-contiguous-match case and the injection-safety case against real corpus content.
- [ ] #4 The P1 eval harness baselines are re-stamped in the same PR if hybrid or keyword numbers move, with before and after numbers included in the PR description.
<!-- AC:END -->
