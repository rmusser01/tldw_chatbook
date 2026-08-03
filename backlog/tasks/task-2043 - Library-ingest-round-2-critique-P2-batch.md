---
id: TASK-2043
title: >-
  Library ingest round-2 critique P2 batch
status: To Do
assignee: []
created_date: '2026-08-03 02:00'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description (the why)

Remaining P2-grade findings from the round-2 dual-agent critique
(snapshot `2026-08-03T01-33-07Z…`, 24/40) not covered by TASK-2041/2042.

## Acceptance Criteria (the what)

- [ ] Corrupt-file extraction failures (e.g. "PDF Extraction Error") are
      classified permanent (no dead-bait Retry) OR Retry copy explains
      what could make a retry succeed; the details view no longer chains
      two "Failed to …" prefixes.
- [ ] "Show details" presents error details in a readable, copyable,
      non-expiring surface (inline expandable row section), not a ~4s
      toast.
- [ ] Rail re-entry either preserves the staged form for the session or
      warns before discarding it (today it silently wipes path, pre-flight
      and metadata — deliberate but destructive).
- [ ] Select fields (PDF engine, Encoding) get visible labels like the
      value inputs did in task-2012.
- [ ] Checkbox on/off is distinguishable without color (custom glyph or
      suffix; stock Textual renders "X" for both states).
- [ ] Collapsed panels no longer carry ~3 blank filler rows; expanded
      panels no trailing blank region.
- [ ] The queue counts line reflects the current batch (or labels itself
      as all-history) so batch outcomes don't blur together.
- [ ] Pre-flight warns "already in your Library" for byte-identical files
      (the content hash exists at pre-flight time) instead of the match
      being an after-the-fact discovery.
- [ ] "Import media" (canvas) vs "Import Media" (picker) casing aligned.
- [ ] Consider: `.md` labeled "plain text file"; a session-persistent
      ingest history surface backed by the existing jobs DB (Clear
      finished currently shreds the only receipts).
