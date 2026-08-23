---
id: TASK-21102
title: >-
  Break the six eager Chunking import chains - ~15k LOC imported at boot to read one string constant
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - imports
  - chunking
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21102).

~15k LOC of `Chunking/` (incl. 28/38 vendored engine modules, a real `import langdetect`, an
nltk `find_spec` path scan, and the Internal_Prompts package) is imported eagerly through SIX
entry points. The first is `Local_Ingestion/local_file_ingestion.py:172` importing
`ENGINE_VERSION` - a string literal (`Chunk_Lib.py:150`). The others:
`Library/ingest_preflight.py:26`, `Library/web_clip_request.py:27`, `RAG_Search/__init__.py:21`,
`RAG_Admin/local_rag_admin_service.py:17-18`, `app.py:1997-2007`. Fixing only app.py buys
nothing - all six must break.

## Acceptance Criteria

- [ ] `import tldw_chatbook.app` no longer executes the Chunking package (nor langdetect) - pinned by a test asserting `"tldw_chatbook.Chunking" not in sys.modules` after importing the app module
- [ ] All six entry points are converted (constant inlined or lazily accessed; PEP-562 re-exports where a facade is needed); chunking behavior at first real use is unchanged
- [ ] Warm `python -X importtime` before/after numbers recorded in the task
