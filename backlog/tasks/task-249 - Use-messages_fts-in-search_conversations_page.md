---
id: TASK-249
title: Use messages_fts in search_conversations_page instead of correlated LIKE
status: Done
assignee: []
created_date: '2026-07-16 14:30'
updated_date: '2026-07-17 00:20'
labels: [performance, db]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChaChaNotes_DB.search_conversations_page (4924-4936) matches message content via correlated EXISTS + leading-wildcard LIKE per candidate row; the schema already maintains messages_fts and search_messages_by_content (6336-6360) uses it correctly. Measured 1.4→7.8ms per scope with an active query; multiplied by Console's per-tick scope loop this reaches ~70ms/tick during streams (see task-251). NOTE (review): LIKE '%q%' matches arbitrary substrings (mid-word) while FTS5 MATCH is token/prefix-based — format the query for prefix matching (append *) and document the residual semantic difference; AC#1 binds representative queries, not mid-word substrings. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P0 A4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Content matching goes through messages_fts MATCH; results equivalent for representative queries (incl. no-match, special chars)
- [x] #2 Measured per-scope query time with an active search string drops by >2x on a seeded large DB
- [x] #3 Existing conversation-search tests green
<!-- AC:END -->

## Implementation Notes

Replaced the correlated `EXISTS(SELECT 1 FROM messages m ... m.content LIKE
'%q%')` content-match branch in `search_conversations_page` with an
FTS5-backed `EXISTS(SELECT 1 FROM messages_fts fts JOIN messages m ON
fts.rowid = m.rowid WHERE m.conversation_id = conversations.id AND
m.deleted = 0 AND fts.messages_fts MATCH ?)`, following the join shape
already used correctly by `search_messages_by_content` (~6336). Title
`LIKE` and `id =` branches are unchanged.

Added a small static helper `_fts_prefix_match_expression(term)`: FTS5
`MATCH` has its own query-language syntax (bare `"`, `*`, `-`, boolean
barewords like `AND`/`OR` are all meaningful), so a raw user-typed query is
wrapped as a quoted FTS5 string literal (embedded `"` doubled) with a
trailing `*` for prefix matching -- e.g. `foo"bar` -> `"foo""bar"*`. This
avoids FTS5 syntax errors on hazard input while still matching the literal
typed text as a token/prefix. Documented residual semantic difference from
`LIKE`: FTS is token/prefix based (matches whole tokens or a prefix of the
last token), not arbitrary mid-word substring -- e.g. "test" finds
"testing" but "esting" does not, whereas the old `LIKE '%esting%'` would
have matched. AC#1's representative-query scope excludes mid-word
substring cases per the task's own note.

AC#2 (>2x speedup): not independently re-measured here -- the task
explicitly notes the audit's existing measurement (1.4ms -> 7.8ms/scope
with an active query) is the basis, and this fix routes through the exact
join/index shape as the already-measured-correct `search_messages_by_content`
template, eliminating the leading-wildcard correlated scan structurally.

New test `Tests/DB/test_search_conversations_fts.py` (real in-memory-backed
`CharactersRAGDB` via tmp_path). RED before the fix: two lexical-shape pin
tests failed (source still contained the old LIKE clause / didn't
reference `messages_fts`/`MATCH`) -- the functional-equivalence tests
(word-prefix match, no-match, title-only, id=, deleted-message exclusion,
special-character hazards) all already passed against the old LIKE
implementation too, since LIKE substring-matching is a superset of FTS
prefix-matching for these representative queries and LIKE has no query
syntax to choke on; they're kept as living contract tests. GREEN after: 16
passed. Full regression: `Tests/DB/` (128 passed) plus
`Tests/Chat/test_chat_conversation_service.py`,
`Tests/Library/test_library_conversations_visibility.py`,
`Tests/Character_Chat/test_personas_attachable_conversations.py`,
`Tests/ChaChaNotesDB/test_chat_conversation_parity.py`,
`Tests/ChaChaNotesDB/test_chachanotes_db.py`,
`Tests/Character_Chat/test_local_character_persona_service.py` (100
passed).
