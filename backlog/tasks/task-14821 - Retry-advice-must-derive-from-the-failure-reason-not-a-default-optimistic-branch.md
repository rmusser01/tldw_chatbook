---
id: TASK-14821
title: >-
  Retry advice must derive from the failure reason, not a default optimistic
  branch
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
updated_date: '2026-08-10 21:41'
labels:
  - library
  - ingest
  - ux
  - copy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 of the 2026-08-10 re-critique. The `Show details` expansion contradicts the row it belongs to and recommends an action that cannot work.

The retry advisory's ELSE branch ("A retry can succeed if the failure was transient — a busy file or a network hiccup…") fires for every failure whose category isn't `parse_error` and whose message doesn't match `_MISSING_DEPENDENCY_RE` (which matches `No module named 'x'` / `x is not installed` / `pip install x`). A missing-OCR failure's own message reads "…install an OCR backend (docling, tesseract, easyocr, paddleocr, or docext)" — which that regex does NOT match — so the optimistic fallback fires. This makes it the COMMON case, not an edge case, and turns the Retry button into a trap for a deterministic failure.

Observed live on two different rows: `✗ failed · diagram.png · No text was found in diagram.png… install an OCR backend (…)` → Show details → `Category: write error` + the transient advisory. Identical on `report_draft.docx`.

Two further defects in the same expansion: `write error` is a mis-category (nothing was written — the failure was extraction producing no content), and `Category: <token>` is a raw internal token shown to users. The expansion also prints the raw ffmpeg banner TWICE (~40 lines including Homebrew Cellar paths and 7 libav version lines) under both `Details:` and `Underlying:`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failure's retry advice is derived from the same reason that produced its row line — a deterministic tooling/extraction failure is never described as possibly transient
- [x] #2 The optimistic advisory appears only for genuinely retryable causes; the unknown case is silent rather than encouraging
- [x] #3 A no-content extraction failure is categorised as such, not as a write error, and the category is user-readable rather than a raw token
- [x] #4 Underlying tool output appears once, not duplicated between Details and Underlying
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Give the two no-content refusals in `_reject_empty_extraction` (local_file_ingestion.py) their own exception classes carrying an `ingest_error_category` ('empty_source' / 'no_content'); they raise BEFORE any write, so 'write error' was never true.
2. `app.py`'s writer failure path reads that category instead of hard-coding `write_error`, so the mis-category disappears at the source.
3. `library_ingest_state.py`: map every category through a user-readable label table for the expansion's first line (raw token retired), and derive the retry advisory from that SAME reason via one `ingest_retry_advice()` helper -- named missing dependency, deterministic tooling/no-content failure (never 'transient'), parse_error corrupt-file advice, genuine write error, and SILENCE for the unknown case.
4. Dedup the underlying chain by containment (whitespace-normalised), not by exact-equality after a `split(': ', 1)` -- the ffmpeg banner survived that split because the row message carries a 'Failed to ingest audio file: ' wrapper the chain entry lacks, so the same ~40 lines printed under both Details and Underlying.
5. Tests: one per advisory branch, plus a mutation check that the optimistic copy cannot reappear for a tooling failure, plus the real ffmpeg-shaped double-banner case.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the mis-category at its source and derived the advisory from the reason.

**Category (AC#3)** — `_reject_empty_extraction` runs at the top of `persist_parsed_media`, BEFORE any write, so 'write error' was never true for it. It now raises `NoContentExtractedError` / `EmptySourceIngestError` (both carrying a class-level `ingest_error_category`), and the app's writer failure path reads `getattr(exc, 'ingest_error_category', 'write_error')` instead of hard-coding one — a genuine DB write failure still reports `write_error`. Verified through the REAL pipeline in the task-14820 governance test: diagram.png lands as `no_content`, empty.txt as `empty_source`.

**User-readable reason (AC#3)** — the expansion's first line is now `Reason: <sentence>` via `ingest_failure_reason()`'s label table (parse_error -> 'The file couldn't be read.', no_content -> 'No text could be extracted.', empty_source, unsupported_file_type, missing_source, write_error, stt_failure). An unmapped token degrades to spaced-out text rather than raising.

**Advisory (AC#1/#2)** — one `ingest_retry_advice(category, message, chain)` helper, consulted only when the row actually offers Retry: a named missing module keeps 'Missing dependency: X. Install it, then Retry.'; a deterministic tooling/no-content failure gets 'Retrying now will fail the same way — install the tooling named above first, then Retry.'; parse_error keeps task-2140's corrupt-file line; write_error keeps the one genuinely retryable optimistic line (scoped to the write, no 'network hiccup'); everything else returns '' — SILENT. The old ELSE branch fired for any category that wasn't parse_error whose message missed `_MISSING_DEPENDENCY_RE`, which the OCR message ('install an OCR backend (docling, tesseract, …)') does — so the optimistic copy was the common case. `_TOOLING_REMEDY_RE` catches that family of remedies that name no importable module.

**Duplicate tool output (AC#4)** — the chain dedup compared exact equality after one `split(': ', 1)`; the real ffmpeg failure walks through it because the message carries a 'Failed to ingest audio file: ' wrapper the chain entry lacks, so the same ~40-line banner printed under both Details and Underlying. Now containment over whitespace-normalised texts (`_restates_known_text`). The RED test is built from a REAL captured `run_parse_job` failure shape (message 1339 chars, chain[0] 1330 chars, job.error = the sanitized first 200) and asserted the banner appeared 2 times before the fix.

Modified: `Local_Ingestion/local_file_ingestion.py`, `app.py`, `Library/library_ingest_state.py`, `Tests/Library/test_library_ingest_state.py`, `Tests/integration/test_library_ingest_flow.py`, `Docs/User_Guide/library/import-and-export.md`.
<!-- SECTION:NOTES:END -->
