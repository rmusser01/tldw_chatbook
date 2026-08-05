---
id: TASK-2220
title: >-
  Library ingest: unsupported-in-folder files record as "skipped", not "failed"
status: Done
assignee: []
created_date: '2026-08-04 05:00'
labels:
  - library
  - ingest
  - ux
priority: high
dependencies: []
---

## Description (the why)

Owner ruling (2026-08-04, round-6 critique follow-up): "failed" is
reserved for files the pipeline TRIED and could not ingest. A folder of
100 photos + 1 PDF currently yields 100 red "✗ failed" rows, a
"1 imported · 100 failed" toast, and 100 permanent failure records —
alarming bookkeeping for entirely normal folder contents. Unsupported
files the user pointed at (via a folder) become a distinct, neutral
"skipped" outcome.

## Acceptance Criteria (the what)

- [x] An unsupported file inside a folder selection reaches a terminal
      "skipped" outcome: neutral glyph/color (not the failure ✗/red),
      no Retry offered, kept in Recent ingests.
- [x] The tally and completion toast count skips in their own segment
      ("1 imported · 100 skipped"), never as failures.
- [x] The pre-flight forecast copy matches the new ontology ("100 will
      be skipped", not "recorded as failures"); the commit summary uses
      "will skip" for them.
- [x] A genuinely attempted-and-failed file (parse error, missing
      source) still records as "failed" with Retry where applicable —
      the skipped state never absorbs real failures.
- [x] Clear finished clears skipped rows with the same confirm; the
      armed label counts them distinctly when present.

## Implementation Plan (the how)

New `IngestJobState.SKIPPED` + `mark_skipped` transition (mirrors the
CANCELLED neutral-terminal pattern); route at the parse-failure choke
point on `category == "unsupported_file_type"`; surfaces follow.

## Implementation Notes

- Jobs: `SKIPPED` enum member; terminal + dismissible sets extended;
  `mark_skipped(job_id, reason, error_detail)` persists and notifies
  like `mark_failed` minus retry semantics (requeue stays FAILED-only).
  DB round-trip is free (`IngestJobState(row["state"])`).
- app.py `_handle_ingest_parse_result`: unsupported_file_type →
  `mark_skipped`; everything else keeps `mark_failed`.
- State: `○ skipped · name · reason` row (no Retry, dismiss offered,
  the CANCELLED template); counts line gains its own `skipped` segment;
  recent/finished include SKIPPED; the armed clear label counts skips as
  finished but only FAILED as "(incl. N failed)".
- Forecast: "N unsupported files will be skipped: …" (was "recorded as
  failures"); commit summary splits "will skip" (unsupported) from
  "will fail" (empty files, which ARE enqueued and genuinely fail).
- Toast: own "N skipped" segment; 4-tuple batch baseline; severity
  warns only when real failures are the batch's only outcome.
- Ledger snapshot at clear time includes SKIPPED.
- Contract pins updated: the two runner tests now assert SKIPPED
  end-to-end (real parse worker), forecast copy pin updated; new pins
  for row rendering/tally/label and the skip-vs-fail commit split.

**Verification.** 289 core + 52 shell-subset targeted green; collect
sweep clean.
