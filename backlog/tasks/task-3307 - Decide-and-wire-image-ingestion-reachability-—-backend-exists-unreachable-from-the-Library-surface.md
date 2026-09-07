---
id: TASK-3307
title: >-
  Decide and wire image ingestion reachability — backend exists, unreachable
  from the Library surface
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
updated_date: '2026-08-09 14:52'
labels:
  - library
  - ingest
  - parity
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-08-07 parity audit: `Local_Ingestion/Image_Processing_Lib.py` implements `process_image` (OCR backend/language, visual features, analysis) but no image extension is mapped in `detect_file_type`, no caller exists in `local_file_ingestion.py` or `Library/`, and the ingest surface's supported list omits images. Either wire the media type (extension mapping → type group → panel → `_ingest_job_options` branch → processor) or record the decision that images stay unshipped and ensure preflight names them as unsupported honestly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An owner decision (ship or defer) is recorded; if ship, image files ingest end-to-end from the Library surface with their options panel
- [x] #2 If deferred, image files are classified as unsupported with honest copy (not silently generic)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read process_image's real signature/return (done: enable_ocr/ocr_backend/ocr_language/extract_features/chunk_options/perform_analysis; returns dict with content=OCR text, chunks, warnings, media_type='image').\n2. RED tests first: detect_file_type mapping (8 raster exts), get_type_group image branch, image TypeGroupCapabilities (ocr toggle + ocr_language + ocr_backend, nouns, warnings via any-of image_ocr probe), _ingest_job_options image branch, parse_local_file_for_ingest wiring (signature-checked process_image stub), end-to-end real 2x2 PNG -> persist_parsed_media -> real MediaDatabase (Pillow IS in this venv), no-OCR-text honesty failure, preflight noun/copy pins, server-mode refusal, options round-trip.\n3. Implement: extension mapping + _ensure_process_image + parse branch (extract_features=False, analysis via the arc's chat_api_call text tail), image schema group + get_type_group branch + any-of feature probe, _TYPE_GROUP_LABELS/SUPPORTED_FORMATS_COPY, app.py image branch (words chunk method), decouple PIL from pillow_heif in Image_Processing_Lib, image-specific empty-extraction message.\n4. Mutation-check the extension mapping and one wiring line; run the six ingest suites + preflight + library_screen + server request suites; update Docs/User_Guide/library/import-and-export.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner ruling: SHIP** (recorded in task-3310's notes: "task-3307 images =
SHIP"). Implemented TDD: a 30-test RED wave across 7 suites preceded the
code (mapping / group schema / app branch / parse wiring / persist path /
copy pins / server refusal / round-trip), and two mutations were run and
Edit-restored: dropping `.png` from `detect_file_type` -> 5 RED across 3
suites; hardcoding the parse branch's `ocr_backend` -> 1 RED wiring test.
Final battery: **780 passed, 1 skipped** (pymupdf, venv-documented) across
test_ingest_capabilities / test_library_ingest_state / test_ingest_preflight
/ test_server_ingest_request / test_library_ingest_runner /
test_submit_library_ingest_job / test_ingest_option_wiring /
test_local_file_ingestion / test_ingest_parse_worker / test_library_screen
/ the six UI ingest suites / the integration flow; the meta-tests
(option_labels / disabled_reason / nouns / gate-value reachability) pass
unmodified. The wider 3-dir sweep (Tests/Library + Tests/App +
Tests/Local_Ingestion) was 1832 passed with 4 pre-existing failures in
modules this task never touched (trafilatura absent x2, GGUF admission,
transcription-context kwargs drift in test_audio_model_dir_routing /
test_transcribe_cpp_ingestion) -- each fails identically in isolation and
none imports image code.

**End-to-end shape** (mirrors the task-3303 document-group recipe):

- `detect_file_type` maps `.png/.jpg/.jpeg/.gif/.webp/.bmp/.tiff/.tif` ->
  `"image"` -- exactly the raster set the processor's PIL loader opens on a
  plain Pillow install. `.svg` (vector, not PIL-rasterizable), `.ico` (icon
  container, not content) and `.heic/.heif` (need pillow_heif, which no
  extra ships) stay honestly unsupported even though process_image's own
  SUPPORTED_IMAGE_FORMATS table lists them. `get_supported_extensions` and
  the unsupported-error copy updated to match.
- `get_type_group` -> new `image` branch BEFORE the URL check (like
  pdf/ebook/document: a `.png` URL is an image, not a page to scrape).
- Schema (`ingest_capabilities`): new `image` TypeGroupCapabilities --
  nouns "image"/"images"; fields `ocr` (checkbox, default True, hint says
  the extracted text is what gets imported), `ocr_language` (rides `ocr`),
  `ocr_backend` (select: auto/docext/docling/tesseract/easyocr/paddleocr --
  the OCR manager's registered backends in its own priority order, every
  option labeled). `required_features=("image_ocr",)`: a NEW any-of
  umbrella (`_FEATURE_ANY_PACKAGES`, consulted by `_probe_installed`)
  that is installed when ANY of docext/docling/pytesseract/easyocr/
  paddleocr imports -- the all-of grammar can't say "one backend,
  whichever", and optional_deps' `ocr_processing` flag reports available
  on a bare `openai` import, so it was deliberately not reused. Recovery
  resolves via the ocr_docext extra with a group hint override
  ("extracting text from images"); label "OCR backend".
- App branch (`_ingest_job_options`): forwards ocr/ocr_language/
  ocr_backend with process_image's own defaults as fallbacks and sets the
  explicit `words` chunk method (F12 unit-honesty parity -- process_image
  chunks the OCR text through the same improved_chunking_process).
- Parse branch (`parse_local_file_for_ingest`, lazy `_ensure_process_image`
  placeholder per the module convention): `process_image(file_path,
  title/author/keywords overrides, enable_ocr=, ocr_backend=,
  ocr_language=, extract_features=False, chunk_options=<dict when chunk ON
  / None when OFF>, perform_analysis=False)`. Returns dict with
  `content`=OCR text, `chunks`, `warnings`, `media_type="image"`; flows
  through the shared tail into `persist_parsed_media` like every type.
- **Analysis decision:** the arc's chat_api_call text tail, NOT the
  processor path -- `image` joined `_TEXT_ANALYSIS_TYPES` and the branch
  forces `perform_analysis=False` at the processor, because
  process_image's internal path is the legacy `Summarization analyze()`
  direct dispatch (dead on a normal install per task-3301) and can carry
  neither the [analysis_defaults] call shape nor keyless dispatch. Pinned
  by a test that explodes if `analyze()` is touched.
- **Visual-features decision: rejected-with-note.** `extract_features` is
  forced off and no toggle exists: the features dict lands in
  `result["visual_features"]`, which the payload does not carry, and
  `persist_parsed_media` forwards no metadata at all -- the control would
  be paid-for compute whose output is dropped end-to-end.
- **No-OCR-text honesty:** the established `_reject_empty_extraction` gate
  fails the job at persist (no empty searchable-in-name-only rows), with a
  new image-specific message ("No text was found in X. An image import
  stores the text OCR extracts; turn Extract text (OCR) on and install an
  OCR backend...") -- the generic copy's "may be scanned images" reads as
  nonsense for an image. Retryable on purpose (installing a backend fixes
  the next attempt). Preflight names the missing backend up front via the
  required `image_ocr` warning (+ install command).
- **Server-mapping decision: NOT mapped.** `SERVER_ACCEPTED_MEDIA_TYPES`
  (established against a live server) has no image type; mapping to
  `document` would ask the server's text extractor to read pixels. Image
  files in server mode raise the existing honest
  `ServerIngestUnsupported` ("no handler for 'image' sources") -- pinned.
- **Duplicate-forecast decision: images stay OUT** of the raw-bytes probe.
  The probe compares sha256(file bytes) against PARSED-content hashes;
  an image's stored content is its OCR text, so raw bytes could never
  match -- same correctness reasoning that dropped .docx from the probe
  in task-3303 (the task prompt's "images are raw bytes end-to-end"
  intuition is wrong at the dedup layer).
- **Heavy-lane decision:** not heavy. `_INGEST_HEAVY_TYPES` stays
  {audio, video}; image OCR is closer to pdf-with-OCR, which also rides
  the normal lane.

**Every-surface audit** (per-surface outcome):
- detect_file_type / get_supported_extensions / error copy -- new mapping.
- get_type_group -- new branch before the URL check.
- `_TYPE_GROUPS` + `_GROUP_EXTRAS` ("image" -> ocr_docext) +
  `_FEATURE_LABELS`/`_FEATURE_TO_EXTRA`/`_GROUP_FEATURE_HINTS` -- new rows.
- preflight noun + intro + SUPPORTED_FORMATS_COPY -- `_TYPE_GROUP_LABELS`
  gains ("image","images") between e-books and plain text; copy updated;
  4 stale pins updated (state copy tail, 2 canvas copy tails, picker
  filter) plus fixtures that used .jpg as the canonical unsupported file
  (preflight x2, runner x1, state renderer fixtures) -> .srt/.xyz.
- panel appearance -- automatic (canvas renders state.type_groups; group
  id has no hyphen so the `opt-` id parser is safe).
- persisted options load/save -- automatic (list_type_groups +
  cap.field_names); new round-trip test drives the real save/load seams.
- server request -- refused honestly (decision above), test-pinned;
  `test_every_mapping_target_is_one_the_server_accepts` untouched.
- picker importable filter -- automatic via get_type_group; pin updated to
  assert .jpg IS importable now and .srt is not.
- capability warnings / guardrail counts -- generic via
  get_capabilities/get_tooling_warnings/_affected_counts; image gets the
  one required `image_ocr` warning with install command.
- heavy lane / queue rows / done progress -- detected_type "image" rides
  the normal lane; queue surfaces are type-string-agnostic.
- duplicate forecast -- stays scoped to `generic` (decision above).
- Home/active-work -- greps found no type enumerations; Media browse type
  filter is built from distinct DB types, so "image" appears automatically.

**Also fixed in passing:** `Image_Processing_Lib`'s import guard coupled
PIL and pillow_heif in ONE try block, so a missing pillow_heif (no extra
ships it) flagged PIL_AVAILABLE=False on every normal install with Pillow
present -- killing metadata, preprocessing and visual features. Split
into independent guards (`PIL_AVAILABLE` / `HEIF_AVAILABLE`); the
end-to-end test pins real-PIL metadata (width/height of a real 2x2 PNG)
through the payload.

**Files:** tldw_chatbook/Local_Ingestion/local_file_ingestion.py,
Image_Processing_Lib.py, tldw_chatbook/Library/ingest_capabilities.py,
library_ingest_state.py, tldw_chatbook/app.py,
Docs/User_Guide/library/import-and-export.md (+ task-3307 stamp). Tests:
test_ingest_capabilities (+7), test_local_file_ingestion (+3),
test_submit_library_ingest_job (+3), test_ingest_option_wiring (+7 incl.
real-PNG end-to-end against a real MediaDatabase), test_library_ingest_state
(+3), test_server_ingest_request (+2), test_ingest_preflight (+1),
test_library_screen (+1 round-trip).

**AC#2 note:** ticked as vacuous -- its condition ("if deferred") never
arose under the ship ruling. The honest-unsupported pattern it asks for
is exercised anyway for the image lookalikes (.svg/.ico/.heic/.heif),
which pre-flight into the unsupported bucket with the supported-list copy.

**Owner-review items:** (1) the `image_ocr` recovery command names the
ocr_docext extra (torch-heavy) because it is the only OCR-purposed
pyproject extra -- docling-via-[pdf] or a bare pytesseract install work
too and the warning hint stays generic; (2) `.heic/.heif` stay unsupported
until someone ships pillow_heif in an extra (the decoupled guard is ready
for it); (3) images are local-only by design -- revisit if the server
ever grows an image media type.
**xhigh review round (2026-08-09):** three confirmed defects in this task's
own new code, all fixed under TDD with READ reds.

1. **The canvas promised OCR on image URLs the pipeline never OCRs.** The
`image` branch of `get_type_group` sat before the URL check (copying the
pdf/ebook/document precedent), so `https://example.com/chart.png` reported
"1 image", mounted the OCR fold, and raised the missing-OCR-backend warning
that forces the new two-press consent — after which the pipeline
(`classify_ingest_source` has no image branch) scraped the URL as HTML and
discarded every OCR option. **Decision: group image URLs as `web`**, not
route them to `process_image`. Downloading and OCR'ing image URLs is a real
feature, but it is one the pipeline would have to grow first (fetch, size
caps, egress policy); until then the canvas tells the truth. The pdf/ebook
precedent does not transfer: their unused options are inert, whereas the
image verdict changes the panel AND costs the user a consent press. The
existing canvas-vs-pipeline agreement test now covers image URLs with NO
compatibility allowance for the `image` group, and the task's own
`test_image_url_extension_wins_over_web_task_3307` was inverted (it pinned
the defect).

2. **Image OCR text was never chunked by the form's settings.** The branch
delegated to `process_image`'s internal chunking, which chunks only for a
TRUTHY `chunk_options`; "Chunk content ON with nothing typed" arrives as
`{}`, and `image` was deliberately absent from `_TEXT_CHUNK_TYPES`, so the
shared tail's repair never ran either — the image persisted as one
unchunked whole-text chunk whatever size the form asked for. **One chunking
authority:** `process_image` is now always called with `chunk_options=None`,
its convenience single chunk is dropped, and `image` joined
`_TEXT_CHUNK_TYPES` so the same tail that chunks plaintext/html chunks OCR
text. Governance test uses the real chunker (only the OCR boundary is
stubbed) and asserts chunk size governs chunk count; mutation-checked by
removing `image` from `_TEXT_CHUNK_TYPES` (4 red).

3. **The OCR-backend probe disagreed with the OCR manager.** The
`_FEATURE_ANY_PACKAGES["image_ocr"]` umbrella re-derived availability from
single import names, so `paddleocr` alone (OCR_Backends needs `paddle` too)
or bare `docext` (needs a gradio_client/transformers/openai companion for
its mode) reported "an OCR backend is installed" — no warning, then an
empty-extraction failure. The table is now ANY-OF-over-ALL-OF and mirrors
OCR_Backends' rules exactly. Importing OCR_Backends from the render path
was rejected: its flags are computed at import time and the module builds
the global `OCRManager` (registering five backends) as a side effect, so a
memoised `find_spec` probe stays the right shape. Instead the guard test
drives the REAL backend classes against each group and each group-minus-one
package, by reloading OCR_Backends under a patched resolver — the old flat
rules make it red on exactly the paddleocr/docext rows. One honest gap
recorded in the test: `TesseractOCRBackend.is_available()` also shells out
for the tesseract binary, which a package probe cannot and should not
replicate.

Files: `Library/ingest_capabilities.py` (URL-aware image branch; any-of-
over-all-of umbrella + `_probe_installed`),
`Local_Ingestion/local_file_ingestion.py` (`_TEXT_CHUNK_TYPES` + image
branch), `Docs/User_Guide/library/import-and-export.md`. Tests:
`test_ingest_capabilities.py` (image-URL rows, inverted 3307 test, local-
image regression, 3 OCR-umbrella guards incl. the pinned backend roster),
`test_ingest_option_wiring.py` (3 image-chunking tests; the
chunk_options-passthrough assertion corrected).
<!-- SECTION:NOTES:END -->
