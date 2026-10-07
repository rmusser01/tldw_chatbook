---
id: TASK-33261
title: 'PERF-02: Conversation search runs its FTS MATCH once per candidate conversation
  (TASK-278 regression)'
status: Done
created_date: 2026-09-28 18:01
labels:
- performance
- database
- search
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
assignee:
- '@claude'
updated_date: 2026-09-28 18:51
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_conversations_page (DB/ChaChaNotes_DB.py ~11759) now filters with a correlated EXISTS(SELECT ... messages_fts MATCH ...). SQLite re-runs the full-text query once per candidate conversation. Measured at 150k messages: about 3 s for a rare term and about 70 s for a common one; the DB bench shows 1,124 ms versus 7 ms for the uncorrelated IN form. This is the TASK-278 rewrite (Done) regressing. It hits Ctrl+K History search, Library > Conversations search and the Console browser search. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-02; every issue with file:line is listed under PERF-02 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation content search evaluates the messages_fts MATCH once per query, not once per candidate conversation
- [x] #2 A plan-pin test captured with sqlite_stat1 absent fails on the correlated form and passes on the fix
- [x] #3 Existing conversation-search behaviour tests pass unchanged (title/id matching, workspace scoping, deleted-message exclusion)
- [x] #4 A seeded benchmark at 3,000 conversations / 150,000 messages shows a rare-term content search under 10 ms (was 219 ms) and a term present in every message under 250 ms (was 13 s) against the correlated form
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: plan-pin tests on the count statement (sqlite_stat1 absent) asserting no CORRELATED subquery for both the single-query and per-term branches, plus a per-term AND-semantics behaviour test.
2. GREEN: replace the correlated EXISTS with one uncorrelated id IN (SELECT m.conversation_id FROM messages_fts ... MATCH ?) clause shared by both branches, keeping the hidden-column MATCH form.
3. Run every test file that calls search_conversations_page plus Tests/ChaChaNotesDB and Tests/DB.
4. Benchmark old vs new shape on a seeded 3,000-conversation / 150,000-message scratch DB.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Replaced the correlated EXISTS(... m.conversation_id = conversations.id AND messages_fts MATCH ?) in _conversation_search_filter with one uncorrelated content_match_clause, id IN (SELECT m.conversation_id FROM messages_fts fts JOIN messages m ON fts.rowid = m.rowid WHERE m.deleted = 0 AND fts.messages_fts MATCH ?). Both the single-query and the per-term (query_terms) branches share it. SQLite now runs the MATCH once and probes the materialized id list. The hidden-column MATCH form is kept (the bare alias fails inside a JOIN).

Tests (Tests/DB/test_search_conversations_fts.py, TestContentMatchRunsOncePerQuery): two plan pins on the count statement, captured with sqlite_stat1 absent, assert no CORRELATED subquery for either branch. They failed first with 'CORRELATED SCALAR SUBQUERY' over the FTS scan. A behaviour test pins AND semantics across query_terms. All 67 tests in the file pass.

Benchmark on a scratch DB with 3,000 conversations and 150,000 messages (medians, count plus page statements):
- rare term: 219 ms -> 4.1 ms
- a term present in every message: 13,075 ms -> 197 ms
Totals are identical. The remaining ~200 ms for an every-message term is the FTS scan plus the rowid join over 150k hits, paid by both the count and the page statement. Merging them into one statement would break the tested count/page snapshot seam, so it is not done.

Regression check: every test file that calls search_conversations_page, plus Tests/ChaChaNotesDB and Tests/DB (3,005 tests), passed except 316 failures.
- 309 of those fail identically at the unchanged base commit 9cd9aad65f (pre-existing on dev in this environment, e.g. CharactersRAGDB has no attribute _db_diagnostic_ref, and _pin_sqlite_source() missing reservation/deadline).
- The other 7 are helper-process timing tests, which pass when re-run in isolation (7 passed).

Files: tldw_chatbook/DB/ChaChaNotes_DB.py, Tests/DB/test_search_conversations_fts.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
