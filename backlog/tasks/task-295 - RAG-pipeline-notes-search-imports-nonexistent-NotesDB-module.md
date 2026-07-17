---
id: TASK-295
title: RAG pipeline notes search imports nonexistent NotesDB module — dead path
status: To Do
assignee: []
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
- [ ] #1 search_notes_fts5 queries the real notes store and returns results end-to-end (unmocked test)
- [ ] #2 A decision is recorded on wiring app.db_config (or replacing the seam) so the pipeline paths are reachable in-app
<!-- AC:END -->
