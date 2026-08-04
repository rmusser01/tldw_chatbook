---
id: TASK-2223
title: >-
  Library ingest rulings small batch (no-op consent line, Recent readability, chunk units, matched receipt, panel naming)
status: Done
assignee: []
created_date: '2026-08-04 05:00'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description (the why)

Owner rulings (2026-08-04, round-6 follow-up), the small items:

1. All-match selections keep Start ENABLED (the dedup probe is capped
   best-effort) but consent becomes informed via the quiet line.
2. Recent ingests is unscannable (full ~130-char absolute paths, no
   times).
3. Chunk size/overlap never state their unit and disclose the 100-5000
   range only via error.
4. "Open in Library" on a matched duplicate lands on the twin with no
   explanation.
5. PDF selections render both "PDF documents" and "Plain text /
   documents / HTML" panels — "documents" collides.

Also recorded: the queue apparatus STAYS (load-bearing for
minutes-long audio/video transcription; batch grouping gives fast runs
their result-log feel) — no de-queue work.

## Acceptance Criteria (the what)

- [x] When the forecast predicts zero imports and ≥1 match, the quiet
      line beside Start reads to the effect of "Everything here appears
      to already be in your Library — starting will re-check and match,
      not re-import." Start stays enabled.
- [x] Recent ingests lists basename + relative time per entry, with the
      full path available (secondary line or expansion), and keeps the
      "(dismissed)" marker.
- [x] Chunk size/overlap labels or placeholders state the unit
      (characters) and the valid range up front.
- [x] The Library media detail view notes when the item was reached via
      a dedup match ("matched an existing item — nothing new was
      imported").
- [x] The generic panel title no longer collides with the PDF panel's
      ("documents" appears once across panel titles).

## Implementation Plan (the how)

Five independent edits: state consent line; Recent two-line rendering
(basename + relative age via `format_console_relative_age`, muted full
path); OptionField.hint rendered in field labels; arrival-note kwarg on
the media-viewer state builder threaded from `_open_job_in_library`'s
dedup check; generic panel rename.

## Implementation Notes

- Consent line: computed with the commit summary (0 import + ≥1 match +
  0 fail → informed-consent copy in `start_quiet_line`); Start stays
  enabled.
- Recent: basename — state (dismissed) · relative age, with the full
  path as a muted second line (`.library-ingest-recent-path`).
  `format_console_relative_age` requires keyword `now` (caught by
  suite).
- Chunk hints: `OptionField.hint` ("characters · 100–5000" / "at least
  0") rendered as a label suffix; receipts stay short (label alone).
  The visible-labels pin relaxed to substring semantics.
- Matched receipt: `_open_job_in_library` sets a one-shot
  `_library_media_arrival_note` when the job's progress carries the
  dedup prefix; the VIEWER-RENDER call site consumes it (first wiring
  hit the chat-handoff builder — wrong site, would have swallowed the
  note silently); `build_library_media_viewer_state(arrival_note=…)`
  renders it as the first metadata line.
- Panel renamed "Plain text & HTML" — "documents" now appears once
  across titles; pins updated.

**Qodo round (fixed in `2fc94f5a8`):** REAL bug — the dedup-note check
ran `str().startswith` against a dict-shaped progress payload (never
matches); now delegates to `count_duplicate_done_jobs`, the tally's own
predicate. Plus import order + Args docstring.
