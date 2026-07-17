---
id: TASK-295
title: RAG pipeline notes search imports nonexistent NotesDB module — dead path
status: Done
assignee: ['@claude']
created_date: '2026-07-17 21:50'
labels: [bug, rag, notes]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during task-260: `RAG_Search/pipeline_functions_simple.py`'s `search_notes_fts5` is broken three ways — (1) `from ...Notes.DB.Notes_DB import NotesDB` resolves beyond the top-level package (flattened-module leftover, same class as the conversations defect fixed in task-260), (2) no `Notes_DB.py` module exists anywhere in the tree (notes live in ChaChaNotes), and (3) its call contract (`NotesDB(path)` + `search_notes(user_id=...)`) doesn't match the real `CharactersRAGDB.search_notes(search_term, limit, fts_match_query)` API. Today the function always returns `[]` before crashing because nothing sets `app.db_config['notes_db_path']` — the whole `db_config` seam is never populated by the app, which also keeps the (now-fixed) conversations path returning `[]` in production. Fixing this properly means deciding how the v2 pipeline should reach the notes store (route through `CharactersRAGDB.search_notes` at the chacha path, or the Library RAG search service) and whether/where `app.db_config` should be wired. Unmocked end-to-end tests required (this file's defects were all invisible to mocked tests).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 search_notes_fts5 queries the real notes store and returns results end-to-end (unmocked test)
- [x] #2 A decision is recorded on wiring app.db_config (or replacing the seam) so the pipeline paths are reachable in-app
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Decide the DB seam: prefer the app's live chachanotes_db instance; keep db_config['chacha_db_path'] as construction fallback for tests/probes; do NOT wire db_config in production.
2. Rewire search_notes_fts5 to CharactersRAGDB.search_notes (notes live in ChaChaNotes; the imported Notes_DB module doesn't exist and its user_id call shape never matched the real API).
3. Route both pipeline searches through one _resolve_chacha_db resolver; unmocked real-DB tests for every seam.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
AC#2 wiring decision, recorded in _resolve_chacha_db's docstring: the pipeline now prefers TldwCli.chachanotes_db — the live instance the app resolves at startup (thread-local connections open, schema checked) — which makes both pipeline paths reachable IN-APP with zero startup changes and no per-search CharactersRAGDB construction. db_config stays as a documented construction seam for tests/probes only; wiring it in production was rejected as a second source of truth for a path the app already owns.

search_notes_fts5 rewired: resolver DB + CharactersRAGDB.search_notes(search_term, limit) (no user_id — that gate came from the phantom Notes_DB API and always returned [] anyway); metadata now carries the real columns (created_at, last_modified) instead of nonexistent updated_at/tags. search_conversations_fts5 switched to the same resolver; its task-260 db_config-seam tests pass unmodified, pinning back-compat.

Tests: Tests/RAG_Search/test_pipeline_notes_search.py — 5, all real-DB unmocked (notes e2e via live seam, notes e2e via db_config seam, conversations via live seam, resolver-preference proof incl. no-fallback-construction, no-seam guard).

Files: RAG_Search/pipeline_functions_simple.py, Tests/RAG_Search/test_pipeline_notes_search.py.
<!-- SECTION:NOTES:END -->
