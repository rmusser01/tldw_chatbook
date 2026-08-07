---
id: TASK-3303
title: >-
  Expose high-value backend ingest options per media type (document OCR, PDF OCR detail, ebook chapters, AV translation+VAD, honest web scope)
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
Finding MI-06 (P1), owner-approved high-value subset, from the 2026-08-07 options-parity audit (full matrix in `.impeccable/critique/2026-08-07-media-ingest-ux-options-review.md`). Backend-supported options with no UI path:

- **Document OCR**: `.docx/.odt/.rtf` land in the generic panel (preflight labels them "plain text file"); `process_document`'s `processing_method`/`enable_ocr`/`ocr_language` are unreachable — scanned Word docs cannot be OCR'd from the UI, and `_ingest_job_options` has no document branch.
- **PDF OCR detail**: `ocr_language` and `ocr_backend` unreachable; `docext` engine missing from the engine select; UI permits Enable-OCR with engines that can't OCR (silent no-op).
- **Ebook chapter chunking**: chunk method is hardcoded "sentences"; the ebook_chapters config default dies in never-used `common_params` — chapter chunking is unreachable.
- **AV translation + VAD**: `translation_target_language` and `vad_filter` are accepted by `process_audio_files` but no UI field ever sets them.
- **Local web scope honesty**: `scrape_method`/`max_pages`/`max_depth` are honored only by the server clip path; a local "sitemap" crawl silently imports one page.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Document files get their own type-group options panel (correct preflight noun) exposing processing method, OCR toggle, and OCR language, wired through to `process_document`
- [ ] #2 PDF panel exposes OCR language and OCR backend (gated to OCR-capable engines) and includes `docext` in the engine select; enabling OCR with an incapable engine is prevented or explained at the control
- [ ] #3 Ebook chunking can produce chapter-based chunks from the UI, and the choice reaches the processor
- [ ] #4 Audio/video panel exposes translate-to-English and VAD filter, wired to the transcription call
- [ ] #5 On the local path, multi-page scrape options are either honored or visibly inert with the reason at the control (no silent single-page import); server path behavior unchanged
- [ ] #6 Every new option round-trips the persisted `[library.ingest_options.<group>]` defaults and is covered by a wiring test asserting the value reaches the backend call
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify each gap on the worktree; read `ingest_capabilities.py` schema grammar (enabled_when, hints) and `_ingest_job_options` group branches.
2. Add a `document` type group (extension mapping + preflight noun + panel) and its wiring branch; add PDF ocr_language/ocr_backend/docext with `enabled_when` gating; ebook chunk-method (or chapters toggle) wired via chunk_options; AV translation+VAD fields; web local-path gating with reason-in-label.
3. Wiring tests per option asserting the real call kwargs (assert against real signatures, never hand-written fakes); persistence round-trip tests; targeted suites.
<!-- SECTION:PLAN:END -->
