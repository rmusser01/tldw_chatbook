---
id: TASK-25704
title: Render paths read DEPENDENCIES_AVAILABLE flags that no probe has populated
status: To Do
assignee: []
created_date: '2026-08-30 16:51'
labels:
  - tech-debt
  - optional-deps
  - correctness
  - critique-2026-08-30
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-24704 (Qodo #4) found the Console Inspect rail reporting 'RAG: Unavailable' on a machine where the extras ARE installed: it read DEPENDENCIES_AVAILABLE['embeddings_rag'] directly, and that entry starts False and is only ever populated by check_embeddings_rag_deps(), which nothing calls automatically. Demonstrated on the dev machine -- the registry flag read False while embeddings_rag_deps_installed() read True.

optional_deps.py already documents the shape of the fix: embeddings_rag_deps_installed() is a cheap find_spec probe with no imports and no registry mutation, explicitly safe for render paths, and lazy_embeddings_rag_available() re-probes a False flag rather than trusting it. Both exist precisely because reading the raw flag was already known to be wrong (task-657 records the same defect in EmbeddingFactory).

The Inspect rail's own reader is fixed. This task is the sweep: 14 other raw reads across 7 modules (Voice_Cloning_Window, lab_speech_status, ingest_capabilities, widget_helpers, Web_Server/serve, Embeddings_Lib, and chat_screen's remaining ones). Each needs checking against whether its key is lazily populated and whether a cheap probe exists for it -- some may be fine because their key is eagerly set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every render-path read of a lazily-populated DEPENDENCIES_AVAILABLE key either probes or is documented as safe to read raw
- [ ] #2 A feature that is installed is never reported as unavailable purely because nothing probed it
- [ ] #3 Any probe added to a render path is cheap and non-mutating, or cached
<!-- AC:END -->
