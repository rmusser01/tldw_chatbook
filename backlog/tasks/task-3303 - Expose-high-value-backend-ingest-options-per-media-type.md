---
id: TASK-3303
title: >-
  Expose high-value backend ingest options per media type (document OCR, PDF OCR detail, ebook chapters, AV translation+VAD, honest web scope)
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
Finding MI-06 (P1), owner-approved high-value subset, from the 2026-08-07 options-parity audit (full matrix in `.impeccable/critique/2026-08-07-media-ingest-ux-options-review.md`). Backend-supported options with no UI path:

- **Document OCR**: `.docx/.odt/.rtf` land in the generic panel (preflight labels them "plain text file"); `process_document`'s `processing_method`/`enable_ocr`/`ocr_language` are unreachable — scanned Word docs cannot be OCR'd from the UI, and `_ingest_job_options` has no document branch.
- **PDF OCR detail**: `ocr_language` and `ocr_backend` unreachable; `docext` engine missing from the engine select; UI permits Enable-OCR with engines that can't OCR (silent no-op).
- **Ebook chapter chunking**: chunk method is hardcoded "sentences"; the ebook_chapters config default dies in never-used `common_params` — chapter chunking is unreachable.
- **AV translation + VAD**: `translation_target_language` and `vad_filter` are accepted by `process_audio_files` but no UI field ever sets them.
- **Local web scope honesty**: `scrape_method`/`max_pages`/`max_depth` are honored only by the server clip path; a local "sitemap" crawl silently imports one page.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Document files get their own type-group options panel (correct preflight noun) exposing processing method, OCR toggle, and OCR language, wired through to `process_document`
- [x] #2 PDF panel exposes OCR language and OCR backend (gated to OCR-capable engines) and includes `docext` in the engine select; enabling OCR with an incapable engine is prevented or explained at the control
- [x] #3 Ebook chunking can produce chapter-based chunks from the UI, and the choice reaches the processor
- [x] #4 Audio/video panel exposes translate-to-English and VAD filter, wired to the transcription call
- [x] #5 On the local path, multi-page scrape options are either honored or visibly inert with the reason at the control (no silent single-page import); server path behavior unchanged
- [x] #6 Every new option round-trips the persisted `[library.ingest_options.<group>]` defaults and is covered by a wiring test asserting the value reaches the backend call
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify each gap on the worktree; read `ingest_capabilities.py` schema grammar (enabled_when, hints) and `_ingest_job_options` group branches.
2. Add a `document` type group (extension mapping + preflight noun + panel) and its wiring branch; add PDF ocr_language/ocr_backend/docext with `enabled_when` gating; ebook chunk-method (or chapters toggle) wired via chunk_options; AV translation+VAD fields; web local-path gating with reason-in-label.
3. Wiring tests per option asserting the real call kwargs (assert against real signatures, never hand-written fakes); persistence round-trip tests; targeted suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

All six ACs shipped, TDD throughout: a 36-test RED wave preceded the
implementation (mapping/schema/app-branch/parse-wiring/state), and three
mutations were run and restored via Edit (drop `.rtf` from
`detect_file_type` → mapping + rtf wiring RED; neuter `get_type_group`'s
document branch → mapping + app document branch + server forwarding RED,
8 tests; hardcode the pdf `ocr_language` pass-through → OCR-detail wiring
RED; disable `build_web_scope_note`'s gate → state + rendered-canvas note
tests RED). Final battery: 1479 passed across Tests/Library, Tests/App,
Tests/Local_Ingestion, the four UI ingest suites, test_library_screen,
the integration flow, and config defaults; the only 3 failures are the
venv-documented pre-existing ones (pymupdf / trafilatura absent — same
list as task-3301's notes).

**AC1 — document group.** `.doc/.docx/.odt/.rtf` now map to a new
`document` type group (`get_type_group`), with panel fields
`processing_method` (auto/docling/native), `ocr` (gated
`enabled_when=processing_method in (auto, docling)`, `depends_on=docling`,
hint "docling method only"), and `ocr_language` (rides `ocr`). Wired via a
new `_ingest_job_options` document branch →
`process_document(processing_method=, enable_ocr=, ocr_language=)`;
fallbacks mirror the processor's own signature defaults (pinned by a
wiring test comparing against `inspect.signature`). The generic group
remains the document files' BASE (the flat merge layers document over
generic), so task-3301's chunk/analyze/encoding still apply — pinned by
`test_document_still_gets_generic_chunking` (real MediaDatabase).
Preflight noun: "Word/Office document(s)" (was "plain text file").

**Every-surface audit for the new group** (the "new bucket must be taught
to every surface" sweep — grepped all consumers of the group keys):
- `detect_file_type` — already said "document"; unchanged.
- `get_type_group` — new branch (before the URL check, like pdf/ebook: an
  extension on a URL still says what the target is).
- `_TYPE_GROUPS` schema + `_GROUP_EXTRAS` — new entries.
- preflight breakdown noun — `_TYPE_GROUP_LABELS` entry; intro lines
  derive from the same table, and `SUPPORTED_FORMATS_COPY` updated to
  match (3 test pins updated).
- panel appearance — automatic (canvas renders `state.type_groups` from
  the preflight grouping); group id has no hyphen, so the `opt-` id
  parser is safe.
- persisted options load/save — automatic (`list_type_groups()` +
  `cap.field_names` loops); round-trip pinned end-to-end.
- server request — `SERVER_MEDIA_TYPE_BY_LOCAL_TYPE` already mapped
  document→document (via `classify_ingest_source`, group-independent);
  what CHANGED is `build_server_ingest_kwargs`'s per-group forwarding:
  document options now travel as extras (`MediaIngestJobSubmitRequest`
  is `extra="allow"`; live-server verification of forwarded extras is
  task-3309's). Pinned by a new server-request test.
- picker importable filter — `get_type_group != UNSUPPORTED`; unchanged
  behavior (documents were already importable), still green.
- capability warnings / guardrail counts — `get_tooling_warnings` and
  `_affected_counts` consume `get_capabilities(group)` generically. The
  document group flags `docling` as its one optional feature (the OCR
  enabler); a group-aware hint override (`_GROUP_FEATURE_HINTS`) makes
  its warning read "needed for scanned-document OCR" instead of the pdf
  extra's "PDF ingestion" blurb (the install command is unchanged).
  Native per-format parsers (python-docx/odfpy/striprtf) are deliberately
  NOT feature-flagged: they are alternatives with no pyproject extra to
  name, and a missing one is reported per job with the package named in
  the failure details.
- heavy-lane gate / queue rows / done-progress — detected_type-based
  ("document" already), not group-based; unchanged.
- duplicate forecast probe — stays scoped to `generic`; document files
  drop OUT of it, which is a correctness gain (it sha256's raw bytes
  against parsed-content hashes — a .docx zip could never match).

**AC2 — PDF OCR detail.** `docext` added to the engine select;
`ocr_language` (text, rides `ocr`) and `ocr_backend` (select, docext
engine only — matches `process_pdf`'s `ocr_backend if parser == "docext"`)
added; the `ocr` checkbox is value-gated to docling/docext with the reason
in its label ("Enable OCR (docling or docext engines only)") — the
silent no-op is now unaskable at the control. The canvas learned to render
checkbox hints into the label (previously hints rendered only on
text/number fields). Wired through the pdf branch →
`process_pdf(ocr_language=, ocr_backend=)`.

**AC3 — ebook chapters.** Verified first: `_ingest_job_options` forces no
method (task-3301) and `process_epub` `setdefault`s `ebook_chapters`, so
the chapters DEFAULT was already live — but no UI choice existed. New
ebook `chunk_method` select (chapters/sentences/words/paragraphs);
"chapters" maps to the real `ebook_chapters` in the ebook branch, others
travel verbatim; untouched forms still set no method (pinned), and the
choice is ignored when chunking is off. A meta-test asserts
`ebook_chapters` exists in `Chunk_Lib.Chunker.chunk_text`'s dispatch.

**AC4 — AV translation + VAD.** `translate_to_english` checkbox →
`translation_target_language="en"` (only when no explicit target is
present — retry overrides stay authoritative), value-gated to
default/faster-whisper with hint "via faster-whisper" (parakeet and
transcribe-cpp reject translation in `resolve_batch_stt_route`; under
"default" the route lands on faster-whisper — pinned). `vad_filter`
checkbox → `options["vad_filter"]` → `vad_use=` on BOTH audio and video
processors (previously reachable only through a chunk_options spelling
nobody set; that spelling remains as fallback).

**AC5 — local web scope honesty.** New pure
`build_web_scope_note(ingest_backend, web_options)` +
`WEB_LOCAL_SINGLE_PAGE_NOTE` in `library_ingest_state.py`; the canvas
renders the note directly under the "What to fetch" select
(`#web-local-scope-note`, always-mounted/display-managed) whenever the
ingest targets local and a multi-page method is selected. Server path
untouched (note suppressed when targeting the server; server request
suite green unchanged). Local multi-page crawling NOT implemented
(owner-deferred, per the task).

**AC6 — round-trip + wiring tests.** End-to-end persistence test drives
the real `_do_submit_ingest` save and
`_load_library_ingest_options_from_config` load with only config I/O
stubbed, covering every new field in all five groups; per-option wiring
tests assert the kwargs at the `parse_local_file_for_ingest`→processor
boundary with stubs signature-checked via `inspect.signature` (the
task-3301 pattern), plus `_ingest_job_options`-level tests in
Tests/App.

**Branch repair in passing:** three tests in
`Tests/UI/test_library_screen.py` were failing at branch HEAD
(`test_do_submit_ingest_persists_options`,
`test_faster_whisper_recovery_handler_uses_explicit_provider`,
`test_switch_is_not_offered_when_the_server_seam_cannot_submit`) —
proven pre-existing by running the HEAD copy of the test file from the
scratchpad (3 failed there too). Cause: `_minimal_ingest_screen` (bare
`object.__new__` helper) drifted behind instance attributes the
3301/3302 siblings started reading (`_library_ingest_preflight_generation`,
`_library_selected_row_id`, `_library_ingest_clear_finished_armed`,
`_library_ingest_expanded_details`, `_library_ingest_recent_ledger`).
Helper reseeded with `__init__`'s defaults; whole file now 26/26.

**Signature discoveries vs the task's assumptions:** none contradicted.
`process_document(processing_method/enable_ocr/ocr_language)` and
`process_pdf(ocr_language/ocr_backend, docext engine)` matched;
`process_audio_files` takes `translation_target_language`/`vad_use`;
`vad_filter` pre-change was only read from `chunk_options` (~lines
763/846), never from `options` — that is the gap AC4 closed. OCR backend
names came from `OCR_Backends.OCRManager._register_backends`
(docling/tesseract/easyocr/paddleocr/docext).

**Files changed:** `tldw_chatbook/Library/ingest_capabilities.py`
(document group, pdf/av/ebook fields, get_type_group branch,
`_GROUP_FEATURE_HINTS`), `tldw_chatbook/Library/library_ingest_state.py`
(noun, supported copy, `build_web_scope_note`), `tldw_chatbook/app.py`
(`_ingest_job_options`: document branch, pdf OCR detail, translate/vad,
ebook method mapping), `tldw_chatbook/Local_Ingestion/
local_file_ingestion.py` (document/pdf kwargs, vad from options),
`tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (checkbox hint
labels, web scope note), `Docs/User_Guide/library/import-and-export.md`
(new controls + task-3303 stamp). Tests: extended
`test_ingest_capabilities.py` (+12), `test_library_ingest_state.py` (+6),
`test_submit_library_ingest_job.py` (+14),
`test_ingest_option_wiring.py` (+10), `test_library_ingest_canvas.py`
(+7), `test_library_screen.py` (+1 round-trip, helper repaired),
`test_server_ingest_request.py` (+1).

**xhigh review round 2 addendum (F9/F11/F12, 2026-08-08).**

- **F9 — stale gated translate value failed whole batches.**
  `_ingest_job_options` forwarded `translate_to_english` without
  consulting its `enabled_when` gate (provider must be
  default/faster-whisper): a value ticked under a translating provider,
  then left stale after switching to transcribe-cpp/parakeet-onnx,
  became `translation_target_language='en'` → `BatchSTTRoutingError` →
  every audio/video job in the batch FAILED at dispatch. The builder now
  consults the schema gate via the new
  `ingest_capabilities.field_gate_open(group, field, values)` (within-form
  `enabled_when` only — deliberately not the `depends_on` packaging gate),
  passing the normalized provider so gate and route agree. **Stale-value
  audit of the other gated fields:** `transcription_model_dir` — already
  guarded in the builder; `transcription_model`/`transcription_precision` —
  only consumed when their gate provider is the one requested (route
  model wins otherwise; precision read only in `_parakeet_route`);
  pdf `ocr`/`ocr_language`/`ocr_backend` and document `ocr` — processors
  consume them only under OCR-capable engines (`ocr_supported`); web
  `max_pages`/`max_depth` — already gated at the consumer
  (`web_clip_request` forwards them only for multi-page scrape methods);
  generic `chunk_size`/`chunk_overlap` — gated by the chunk toggle in the
  builder itself. Translate was the only live behavior-changing hazard.
  Mutation: dropping the gate consult → 2 RED.
- **F11 — requeued legacy ebook jobs silently switched chunking scheme.**
  The old builder forced `method='sentences'` for every group; the
  round-1 builder set a method only when the ebook `chunk_method` option
  was present, so a LEGACY persisted snapshot (predating the field) fell
  through to `process_epub`'s chapters default on retry/requeue. Now:
  absent `chunk_method` + chunk ON → builder sets `sentences` (pre-branch
  parity), and `_build_ingest_options_snapshot` seeds the schema default
  ("chapters") into every NEW snapshot so absence unambiguously means
  legacy — a fresh untouched submission still chunks by chapter
  (`ebook_chapters`). The seed also persists to the config echo
  (`library.ingest_options.ebook.chunk_method`), which is the schema
  default anyway.
- **F12 — chunk-size hint claimed words; pdf/audio chunked in sentences.**
  Verified both consumers support a words method (chunking_service's
  in-process words leg for pdf; `ChunkingService.chunk_text` via
  `AudioProcessor._chunk_text` for audio/video), then made the builder
  ALWAYS set an explicit method: pdf + audio/video → `words` (the unit
  the generic hint promises; the processors' sentences setdefaults were a
  ~10-30x unit lie), ebook → its mapped option or the F11 sentences
  fallback, text tail already words. No hint copy change needed — the
  copy is now true; `Docs/User_Guide/library/import-and-export.md` got a
  round-2 "Verified against" stamp. Governance evidence lives in
  task-3301's addendum (real `process_pdf` chunk tail, mutation-checked).
