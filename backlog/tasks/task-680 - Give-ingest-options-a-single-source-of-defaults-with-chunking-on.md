---
id: TASK-680
title: Give ingest options a single source of defaults with chunking on
status: Done
assignee: []
created_date: '2026-07-26 03:26'
updated_date: '2026-07-26 04:25'
labels:
  - ingest
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ingest option defaults are declared in two places that disagree, and the two ingest surfaces ship opposite chunking defaults. Chunking is off in the Library canvas, so imported documents are not chunked for retrieval, which quietly undermines search and RAG for anyone who never opens the advanced panel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Option defaults come from one declaration used by every ingest surface
- [x] #2 Chunking defaults to on with a chunk size of 1000
- [x] #3 Analysis defaults to off so ingest does not make LLM calls unless asked
- [x] #4 A user who previously saved option values keeps them
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The capability schema declared analyze=True/chunk_size=1000 while the form hard-coded analyze=False/chunk=False/chunk_size=500, and the second ingest surface defaulted chunking the other way -- so which defaults a user got depended on which screen they opened.

The capability schema is now the single declaration and the form derives from it. Per the agreed product call, chunking is on with a size of 1000 (local, cheap, and without it imported documents are never chunked for retrieval) while analysis stays off (an LLM call per document that the user has not asked for). Previously saved option values still win, since they are loaded over the defaults.

Changed: tldw_chatbook/Library/ingest_capabilities.py, tldw_chatbook/Library/library_ingest_state.py, Tests/Library/test_library_ingest_state.py, Tests/UI/test_library_ingest_canvas.py
<!-- SECTION:NOTES:END -->
