---
id: TASK-259
title: Console transcript/stream polish (signature cache, buffer collapse, targeted RAG-launch card, inspector rows)
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, console]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four bounded improvements from the audit: _transcript_rows re-derives every row per changed tick (measured 0.86ms@200 msgs → 5.15ms@1000; cache per-message signatures keyed on id/content-len/status); _materialize_stream_buffer re-joins the whole chunk list per tick (collapse after join); _stage_console_library_rag_launch recomposes the ENTIRE ChatScreen for one pending card (5729-5731 — make it a targeted widget; several _build_console_*_state builders read _pending_console_launch_context, so verify); ConsoleRunInspector per-row updates instead of wholesale recompose. Depends on task-251 landing first (shared code region). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P3 D1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Transcript row derivation is O(changed messages) per tick with correctness preserved for delete/reorder/variant-switch
- [ ] #2 Library-RAG launch no longer recomposes the screen
- [ ] #3 Inspector updates changed rows only
- [ ] #4 Streaming behavior verified live (rig)
<!-- AC:END -->
