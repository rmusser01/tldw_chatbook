---
id: TASK-2223
title: >-
  Library ingest rulings small batch (no-op consent line, Recent readability, chunk units, matched receipt, panel naming)
status: To Do
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

- [ ] When the forecast predicts zero imports and ≥1 match, the quiet
      line beside Start reads to the effect of "Everything here appears
      to already be in your Library — starting will re-check and match,
      not re-import." Start stays enabled.
- [ ] Recent ingests lists basename + relative time per entry, with the
      full path available (secondary line or expansion), and keeps the
      "(dismissed)" marker.
- [ ] Chunk size/overlap labels or placeholders state the unit
      (characters) and the valid range up front.
- [ ] The Library media detail view notes when the item was reached via
      a dedup match ("matched an existing item — nothing new was
      imported").
- [ ] The generic panel title no longer collides with the PDF panel's
      ("documents" appears once across panel titles).
