---
id: TASK-15810
title: >-
  Library RAG Answer's first query on a fresh profile never returns — CPU-bound,
  reproduced twice
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-13 20:28'
labels:
  - rag
  - library
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of TASK-15400 (2026-08-12) and again of TASK-15700 (2026-08-13) both stalled at the same place: the FIRST Library RAG Answer run on a freshly-created profile sits on 'searching · <source>…' at ~98% CPU and does not produce an Evidence row. 15400 recorded 4+ minutes; 15700's run was left for 8+ minutes on a 36-note library (36 real User Guide pages written through add_note and indexed through index_entries, embedding model already on disk, HF_HUB_OFFLINE=1 so no download is possible) and still had not rendered a row. A 3-second macOS 'sample' of the process shows the hot stack entirely inside the CPython interpreter (coroutine step -> filter -> set membership -> id()/PySys_Audit), i.e. CPU-bound Python rather than blocked I/O; 'sample' is C-level only, so the Python frame could not be identified from it and the cause is NOT yet attributed. The user-visible effect is that the app's headline retrieval surface appears to hang on first use. Both arcs' live checks had to fall back to driving the engine directly, so this also blocks live verification of any retrieval change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The first RAG Answer run on a fresh profile renders Evidence rows in a bounded, stated time (or discloses progress honestly if a one-time warm-up is unavoidable)
- [ ] #2 The spin is attributed to a named Python frame with a profile (py-spy/cProfile), not to 'the embedding stack' by assumption — the 2026-08-13 sample contradicts that attribution
- [ ] #3 A live retrieval verification on a scratch profile can complete end-to-end through the UI, so a future arc's live check does not have to fall back to the engine
<!-- AC:END -->
