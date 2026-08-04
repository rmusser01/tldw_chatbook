---
id: TASK-2221
title: >-
  Library ingest: batch-grouped queue with latest-batch-first tally
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

Owner ruling (2026-08-04): the queue answers "what did THIS run just
do". Today the tally is a lifetime "N done — all ingests" and two
interleaved batches merge into one flat list. Jobs already carry a
batch_id; the queue groups rows under per-submission headers and the
tally leads with the latest batch, lifetime total secondary.

## Acceptance Criteria (the what)

- [ ] Queue rows group under a per-submission header carrying the
      batch's source (folder/file basename), file count, start time, and
      outcome counts.
- [ ] The tally line leads with the most recent batch's outcome and
      shows the lifetime total secondarily.
- [ ] Two interleaved batches render as two groups; a single-file
      submission reads naturally (no ceremony regression for the
      fast-path).
- [ ] Existing in-place update paths (job ticks, retry, dismiss, clear)
      keep widget identity and scroll behavior within the grouped
      layout.
