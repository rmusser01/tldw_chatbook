---
id: TASK-3301
title: >-
  Wire the dead ingest controls — Analyze, Chunk toggle, Encoding — for real
status: Done
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
- [x] #1 With Analyze on and a configured analysis provider, a local ingest produces a stored analysis; with no provider configured the UI says so before Start (no silent no-op)
- [x] #2 Chunk OFF results in no chunking for pdf/ebook/audio/video paths; Chunk ON results in chunks for plaintext/html/document ingests, with the form's size/overlap governing them — proven by end-to-end tests through the real DB
- [x] #3 The Encoding selection changes how plaintext/html bytes are decoded (test with a latin-1 fixture that mojibakes under utf-8)
- [x] #4 An untouched form submits with the same overlap the UI displays, local and server paths agree, and chunk_size arrives typed as int
- [x] #5 No existing ingest test regresses (targeted Library/Ingestion suites green with nonzero pass counts)
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

## Implementation Notes

All three dead controls are wired for real on the local path, plus the
defaults unification. TDD throughout: 18 characterization tests were RED
against the expected behavior before any implementation (6 more pinned
already-correct behavior), and every new guard was mutation-checked
(8 mutations, each confirmed RED then restored via Edit).

**Chunking mechanism decision (the investigated fork).** There is NO
deferred chunking pass: `Media.chunking_status` is written by
`add_media_with_keywords` but consumed nowhere in the app (only echoed by
`tldw_api/media_reading_schemas.py`), and the DB layer logs-and-ignores
`chunk_options` as an explicit placeholder. The RAG indexing pass
(`RAG_Search/ingestion_indexing.py`) does its own chunking and never reads
`UnvectorizedMediaChunks`. So text types (plaintext/html/document/article)
are chunked **at ingest time in the parse worker** — mechanism (i) — using
`RAG_Search/chunking_service.improved_chunking_process` (the same service
`process_pdf` uses), stored through the existing
`persist_parsed_media -> add_media_with_keywords(chunks=...)` path into
`UnvectorizedMediaChunks` (which IS consumed: `local_media_reading_service`,
MCP resources). `chunking_status` lands `completed`, same as the PDF path.

Two consequences worth knowing:
- `chunk_options is None` now means "no chunking" for EVERY type
  (`parse_local_file_for_ingest` derives `perform_chunking` before the
  `{}` defaulting; the pdf/ebook/audio/video processors get the real
  toggle instead of a hardcoded `True`, and a processor's
  chunking-disabled single-full-text-chunk fallback is dropped from the
  payload so OFF stores zero chunk rows). The one live programmatic
  caller (`local_media_reading_service`) passes `{}`, which stays
  "chunk with defaults"; `quick_ingest`/`batch_ingest_files` defaults
  flip to no-chunking, per the owner's ruling.
- `_ingest_job_options` no longer forces `method: "sentences"`: every
  method in the chunking stack sizes in its OWN unit (sentences method =
  sentence COUNT — form size 1000 meant "1000 sentences", i.e. one chunk,
  which is how the size was dead even where chunking ran). Now pdf keeps
  its sentences default, ebook its chapters default, and the text tail
  uses the service's word method; the schema hint was corrected from
  "characters" to "words" (nothing pinned the old copy, which described a
  pipeline that never chunked).

**Analysis provider seam reused.** New `Library/ingest_analysis.py`
(`resolve_ingest_analysis_provider`) composes the two incumbents:
`[analysis_defaults] provider` (the Media viewer's own analysis default)
resolved through `Chat/provider_readiness.get_provider_readiness` — the
single shared definition of "ready" (config table + env var + placeholder
rejection + keyless locals). `_ingest_job_options` uses it when Analyze is
requested: ready → `api_name`/`api_key` travel to the parse worker (the
processors' existing analysis code runs); not ready →
`analysis_skipped_reason` travels instead. The parse tail analyzes
plaintext/html/article with `Summarization_General_Lib.analyze` (one call
over the full content, the `process_document` pattern; failures are
payload warnings, never job failures); `process_document`'s own summary is
now surfaced (it reported under `summary`, which the tail dropped). The
PDF/ebook processors' `and api_key` analysis gates were relaxed to the
lenient `api_name`-only gate the audio and document processors already
use, so keyless-ready local providers (ollama, ...) analyze pdf/ebook too
— safe because `local_file_ingestion` is those functions' ONLY in-repo
caller and it only sets `api_name` when the resolver said ready
(untestable in this venv: pymupdf/ebooklib absent, so the gates cannot be
executed here; the relaxation matches the audio/document precedent).

**Surfaces taught the analysis-skipped state** (the "every aggregating
surface" audit):
- Ingest canvas, pre-Start: new `#library-ingest-analysis-hint` line
  (always-mounted, display-managed, updated in place by the gate updater;
  the pure state builder gates it on the Analyze toggle). It informs, and
  deliberately does NOT disable Start — analysis is optional.
- Queue done row: progress sub-line reads
  "Imported name — analysis skipped: <reason>" plus a machine-readable
  `progress["analysis_skipped"]` (new pure helper
  `app._library_ingest_done_progress`, unit-tested). Persisted with the
  job, so it survives restarts.
- Details expansion: NOT taught — it renders `error_detail` and is
  failure-only by design; a done row's annotation lives on its progress line.
- Batch header / queue counts / completion toast: NOT taught — they
  aggregate OUTCOME counts (imported/matched/skipped/failed) and analysis
  skip is a per-row annotation, not an outcome state.
- Recent imports fold: NOT taught — deliberately compact
  (name — state · age, task-2223); the queue row carries the note.

**Defaults & typing.** Overlap fallback unified on the schema default via
new `ingest_capabilities.generic_option_default` (the two private
`_generic_default` copies in `library_ingest_state`/`server_ingest_request`
now delegate to it); an untouched form submits overlap 100 on local AND
server paths (pinned by a test comparing both). `chunk_size`/`chunk_overlap`
are coerced to ints at the snapshot boundary
(`_build_ingest_options_snapshot`) and defensively again in
`_ingest_job_options` (persisted config/restored jobs may hold display
strings); chunk_options now carries both `size` (audio/video option maps)
and `max_size` (`improved_chunking_process`) spellings.

**Encoding.** `_decode_ingest_text` in `local_file_ingestion.py`: explicit
selection → that codec with `errors="replace"` (wrong choice degrades
visibly instead of failing); `auto` → strict utf-8, then chardet detection
(the repo's incumbent detector, lazily imported and optional), then
utf-8-with-replace. HTML used to open with STRICT utf-8, so a latin-1 HTML
file failed the entire job — now it decodes per selection.

**Server path**: untouched except the shared-default delegation (its
tests all pass unchanged).

**Files changed**: `tldw_chatbook/app.py` (`_ingest_job_options`,
`_library_ingest_done_progress`, writer), `Local_Ingestion/
local_file_ingestion.py` (toggle/encoding/text chunk+analysis tail),
`Local_Ingestion/ingest_parse_worker.py` (schema doc),
`Library/ingest_analysis.py` (new), `Library/ingest_capabilities.py`
(accessor + words hints), `Library/library_ingest_state.py` (hint state),
`Library/server_ingest_request.py` (delegation), `UI/Screens/
library_screen.py` (snapshot coercion, hint pass, gate updater),
`Widgets/Library/library_ingest_canvas.py` (hint line),
`Docs/User_Guide/library/import-and-export.md` (+ task-3301 stamp).
Tests: new `Tests/Local_Ingestion/test_ingest_option_wiring.py` (24, real
tmp fixtures + real MediaDatabase, processor stubs signature-checked with
`inspect.signature` against the real seams) and
`Tests/Library/test_ingest_analysis.py` (6); extended
`Tests/App/test_submit_library_ingest_job.py` (+13),
`Tests/Library/test_library_ingest_state.py` (+4),
`Tests/UI/test_library_ingest_canvas.py` (+2),
`Tests/integration/test_library_ingest_flow.py` (+1).

**Known pre-existing, unrelated failures in this venv** (not regressions;
they are import/find_spec probes of packages absent from the venv):
`test_web_article_ingestion.py` (2, trafilatura missing),
`test_ingest_capabilities.py::test_installed_feature_produces_no_tooling_warning`
(pymupdf missing), `test_transcription_service_parakeet_buffer_wav.py`
(collection error, numpy missing).

**Two branch repairs beyond 3301's own scope, done in passing:**
- `test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story`
  pinned the pre-task-3302 footer tail ("F6 next pane") and was red on
  this branch's HEAD (task-3302, the previous commit, made the footer
  append the shared global hints but missed this one test). Repaired to
  build its expectation from `AppFooterStatus.GLOBAL_HINTS` so it can't
  drift again.
- `test_library_shell.py::test_library_shell_note_save_result_after_switch_is_discarded`
  is nondeterministic INDEPENDENT of this diff — measured per the
  lessons-testing-evidence protocol: 2 failed / 1 passed with my
  library_screen changes disabled (control), 1 failed / 2 passed with
  them active. The failure is a `#library-note-meta` NoMatches after a
  polled canvas switch (rendering-timing race in the Notes canvas). Left
  as-is; same family as the task-3025 finding.
