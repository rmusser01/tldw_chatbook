---
id: TASK-3301
title: >-
  Wire the dead ingest controls — Analyze, Chunk toggle, Encoding — for real
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - parity
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finding MI-02 (P0) + MI-10 of the 2026-08-07 Media Ingestion review (tracking file `.impeccable/critique/2026-08-07-media-ingest-ux-options-review.md`, parity matrix section). Three options the ingest canvas advertises are silent no-ops on the local path, and the owner ruled they must be wired for real, not annotated away:

1. **Analyze after ingest** never runs locally: `_ingest_job_options` never supplies `api_name`/`api_key` (admitted in its docstring, app.py) and every processor gates analysis on them; plaintext/html/article hardcode `analysis: ""`.
2. **Chunk content** is dead in both directions: OFF is overridden by hardcoded `perform_chunking=True` for pdf/ebook/audio/video in `local_file_ingestion.py`; ON never chunks text types because `add_media_with_keywords` ignores `chunk_options` as a placeholder (`Client_Media_DB_v2.py` ~4065). Investigate the `chunking_status` deferred-chunking column before choosing the text-type mechanism.
3. **Encoding** select is consumed nowhere — utf-8 hardcoded for plaintext/html reads, `errors="replace"` masking wrong choices as mojibake.

Plus **chunk_overlap default mismatch**: UI renders 100, untouched local submit falls back to 50, server paths use 100; and `chunk_size` can reach the backend as a display string.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 With Analyze on and a configured analysis provider, a local ingest produces a stored analysis; with no provider configured the UI says so before Start (no silent no-op)
- [ ] #2 Chunk OFF results in no chunking for pdf/ebook/audio/video paths; Chunk ON results in chunks for plaintext/html/document ingests, with the form's size/overlap governing them — proven by end-to-end tests through the real DB
- [ ] #3 The Encoding selection changes how plaintext/html bytes are decoded (test with a latin-1 fixture that mojibakes under utf-8)
- [ ] #4 An untouched form submits with the same overlap the UI displays, local and server paths agree, and chunk_size arrives typed as int
- [ ] #5 No existing ingest test regresses (targeted Library/Ingestion suites green with nonzero pass counts)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize current behavior with RED end-to-end tests per control (real in-memory DB, real parse path).
2. Analyze: resolve provider/key at job-build time from the app's configured analysis defaults; thread through `_ingest_job_options`; readiness hint in the panel when unconfigured.
3. Chunk: pass `perform_chunking` through to pdf/ebook/av processors; for text types, chunk at ingest (or wire the deferred pass) so the DB stores form-governed chunks; decide via the `chunking_status` investigation.
4. Encoding: consume the option in the plaintext/html readers; auto = current detection/utf-8.
5. Defaults: single source for overlap (100), int coercion at the snapshot boundary.
6. Mutation-check every new guard; run targeted suites.
<!-- SECTION:PLAN:END -->
