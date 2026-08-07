---
id: TASK-3306
title: >-
  Expose remaining audio/video ingest tunables: time-range trim, URL cookies, recursive summary, adaptive/multi-level chunking
status: To Do
assignee: []
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - parity
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred remainder of the 2026-08-07 options-parity audit (matrix in `.impeccable/critique/2026-08-07-media-ingest-ux-options-review.md`; owner scoped the high-value subset to task-3303). `process_audio_files`/`process_videos` accept, and the Library UI cannot reach: `start_time`/`end_time` trim, `use_cookies`/`cookies` for gated URL downloads, `summarize_recursively`, and the adaptive/multi-level chunking + `chunk_language` keys the pipeline reads from `chunk_options` but the app never populates. Also capped whisper model list (no large-v3/distil/turbo) and the permanently-closed `parakeet_defaults_enabled` promotion gate (no production caller).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Each listed tunable is either exposed in the audio/video panel and wired to the processor call, or explicitly rejected in this task's notes with the reason recorded
- [ ] #2 Any exposed option round-trips persisted defaults and has a wiring test against the real call signature
- [ ] #3 Whisper model choices cover the models the routing layer actually accepts
<!-- AC:END -->
