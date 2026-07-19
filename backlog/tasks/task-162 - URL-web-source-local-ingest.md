---
id: TASK-162
title: URL / web source local ingest
status: Done
assignee: []
created_date: '2026-07-11 22:02'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Local ingest currently accepts file paths only. Add URL/web ingest (article extraction / download) as a source for the Library ingest queue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can ingest a URL into the Library
- [x] #2 Web extraction routes through the existing parse/persist pipeline
<!-- AC:END -->

## Implementation Notes

Delivered across four tasks (classifier -> extractor -> parse routing -> submit/UI wiring):

- **Classifier** (`Local_Ingestion/local_file_ingestion.py:classify_ingest_source`): for an http/https URL, returns `"video"` for known video hosts (YouTube, Vimeo, Dailymotion) or a video-extension path, `"audio"` for an audio-extension path, else `"article"`; for any non-URL source it delegates straight to the existing `detect_file_type`. This single seam is reused everywhere a source needs a `detected_type` -- including the heavy-lane cap, which now applies to media URLs for free (no separate URL-aware cap logic needed).
- **Article extractor** (`Local_Ingestion/web_article_ingestion.py`): sync, dependency-light `httpx` (fetch) + `trafilatura` (extract) pipeline with no browser -- runs inside the existing parse-pool worker thread unchanged. Heavy imports are deferred inside the function body so importing the module stays cheap.
- **Parse routing** (`Local_Ingestion/local_file_ingestion.py:ingest_local_file`): an http/https source skips the file-path machinery entirely (no `Path(...).exists()` check) and branches on `classify_ingest_source`'s result -- `"article"` goes through the new extractor and gets a canonicalized URL (`canonicalize_url`: lowercased scheme/host, default port and fragment dropped, tracking params stripped, remaining query params sorted) as its stored `source_url`; `"audio"`/`"video"` pass the raw URL straight through to the existing URL-accepting audio/video processors (server-lesson borrow: those processors already accepted a URL via yt-dlp, so no new download path was needed).
- **Submit path** (`app.py:submit_library_ingest_job`): swapped `detect_file_type` for `classify_ingest_source` so a URL yields a real `detected_type` instead of the classifier throwing `FileIngestionError` and silently degrading to light lane; the `try/except FileIngestionError -> ""` guard is preserved so an unclassifiable *file* still degrades to light instead of crashing.
- **UI validation** (`UI/Screens/library_screen.py:_submit_library_ingest_form`): branches on `urlparse(raw_path).scheme in ("http", "https")` -- a URL goes through `Utils/input_validation.py:validate_url` (syntax check only, no filesystem/network access) and is submitted as-is; anything else keeps today's `validate_path_simple(..., require_exists=True)` behavior unchanged. Placeholder copy on `#library-ingest-path` (`Widgets/Library/library_ingest_canvas.py`) updated to mention a URL.

Modified/added files: `tldw_chatbook/app.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Widgets/Library/library_ingest_canvas.py`, `Tests/UI/test_library_url_ingest_submit.py` (new regression guards on the `classify_ingest_source`/`validate_url` seams).
