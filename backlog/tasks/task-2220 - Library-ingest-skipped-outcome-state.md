---
id: TASK-2220
title: >-
  Library ingest: unsupported-in-folder files record as "skipped", not "failed"
status: To Do
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

- [ ] An unsupported file inside a folder selection reaches a terminal
      "skipped" outcome: neutral glyph/color (not the failure ✗/red),
      no Retry offered, kept in Recent ingests.
- [ ] The tally and completion toast count skips in their own segment
      ("1 imported · 100 skipped"), never as failures.
- [ ] The pre-flight forecast copy matches the new ontology ("100 will
      be skipped", not "recorded as failures"); the commit summary uses
      "will skip" for them.
- [ ] A genuinely attempted-and-failed file (parse error, missing
      source) still records as "failed" with Retry where applicable —
      the skipped state never absorbs real failures.
- [ ] Clear finished clears skipped rows with the same confirm; the
      armed label counts them distinctly when present.
