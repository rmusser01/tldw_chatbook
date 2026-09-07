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
**xhigh review round (2026-08-10)** — the advice was right about the cases it was written for and wrong about four it was not.

*The optimistic branch survived on another path (AC#2).* `app.py`'s write stage stamped `category="write_error"` as the DEFAULT for every exception `persist_parsed_media` re-wraps — and `write_error` is the one category that still earns "a retry can succeed if the write failure was temporary — the file itself parsed fine". So the branch this task removed stayed reachable for every unclassified cause, through the default. `_library_ingest_write_failure_category(exc)` now returns the exception's declared `ingest_error_category`, `"write_error"` only for a genuine DB write failure (`MediaDatabaseError`/`MediaInputError`/`sqlite3.Error`), and `""` otherwise — an unknown cause is unnamed, and an unnamed category is silent.

*A transient failure was sentenced to never-retry.* `_TOOLING_REMEDY_RE`'s `is (?:not|un)available` alternative was tested before the category branches, so `TranscriptionError("The shared local executor is unavailable.")` — a pool teardown that clears on the next attempt — rendered "Retrying now will fail the same way — install the tooling named above first", naming no tooling anywhere. That alternative is replaced by the `requested, but X is unavailable` shape, which only the deliberate-backend refusals raise ("Docling processing requested, but Docling is unavailable").

*The advice cited tooling that was never named.* The generic extraction refusal ("…or the tooling for this file type may not be installed") matched `may not be installed` and got the install-the-tooling-named-above sentence, with nothing above naming anything. The regex no longer matches hedged, subject-less phrasings, and the deterministic categories are now handled SEPARATELY from named remedies: a `no_content`/`empty_source` failure with no remedy on screen says "Retrying now will fail the same way — this file's content, or the tooling for it, has to change first."

*The chain dedup discarded the root cause (AC#4).* `_restates_known_text` tested containment BOTH ways, so a chain entry that quotes the row summary AND appends the underlying cause — the exact diagnostic the chain exists to surface — was dropped as a "restatement". Only `candidate in text` is tested now (a strict superset is kept); the wrapper drift that made equality insufficient in the first place is handled by stripping one `Failed to <verb> <type> file:` stage prefix inside `_normalized_detail_text`, for COMPARISON only. The duplicate-banner fix stays fixed, pinned by its own test.

*Two live-verified fixes to the remedy itself.* Running `run_parse_job` against real fixtures on this install showed the pdf failure's message is `'NoneType' object has no attribute 'FileDataError'` while the remedy lives two links down the chain — so `_TOOLING_REMEDY_RE` is now applied to message AND chain (the chain entries render directly above the advice, so "named above" stays true), matching what `_missing_dependency_from` already did. The same capture showed `pip install (\S+)` swallowing the sentence's full stop: "Missing dependency: tldw_chatbook[pdf].. Install it, then Retry." Package names never carry sentence punctuation, so the capture is right-stripped.

Mutation-checked (F4, F6). Modified: `Library/library_ingest_state.py`, `app.py`, `Tests/Library/test_library_ingest_state.py`, `Tests/App/test_submit_library_ingest_job.py`, `Docs/User_Guide/library/import-and-export.md`.
<!-- SECTION:NOTES:END -->
