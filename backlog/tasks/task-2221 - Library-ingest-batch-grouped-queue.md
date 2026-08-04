---
id: TASK-2221
title: >-
  Library ingest: batch-grouped queue with latest-batch-first tally
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

Owner ruling (2026-08-04): the queue answers "what did THIS run just
do". Today the tally is a lifetime "N done — all ingests" and two
interleaved batches merge into one flat list. Jobs already carry a
batch_id; the queue groups rows under per-submission headers and the
tally leads with the latest batch, lifetime total secondary.

## Acceptance Criteria (the what)

- [x] Queue rows group under a per-submission header carrying the
      batch's source (folder/file basename), file count, start time, and
      outcome counts.
- [x] The tally line leads with the most recent batch's outcome and
      shows the lifetime total secondarily.
- [x] Two interleaved batches render as two groups; a single-file
      submission reads naturally (no ceremony regression for the
      fast-path).
- [x] Existing in-place update paths (job ticks, retry, dismiss, clear)
      keep widget identity and scroll behavior within the grouped
      layout.

## Implementation Plan (the how)

Mint one `local-<uuid>` batch id per folder expansion (threaded through
`submit_library_ingest_job` → `registry.submit`; the job field + DB
column already existed for server batches); group state-side; render
headers + a leading latest-batch line in the queue panel.

## Implementation Notes

- `build_ingest_queue_groups(jobs)` groups CONTIGUOUS same-batch runs
  (row order untouched — submission order keeps runs contiguous, so
  identity/scroll semantics are undisturbed); batchless jobs are bare
  singleton groups, so single-file submissions read exactly as before.
- Header: `▸ {source dirname} — {n} files · {age} · {outcome tallies}`
  (age from the newest member's `finished_at_wall`, "running" while
  active; outcome tallies reuse the counts vocabulary incl. skipped).
- `latest_batch_line` ("Latest batch: …") renders ABOVE the lifetime
  tally — latest first, lifetime secondary, per the ruling.
- Rendering is header-Statics interleaved before each headed group's
  first row — no containers, so the in-place update paths (job ticks,
  retry, dismiss, clear, arming) are untouched.

**Verification.** 293 core + 51 shell-subset green; collect clean.
Live: mixed_folder run shows "2 will import · 2 will skip" → header
"▸ mixed_folder — 4 files · now · 2 done · 2 skipped" with
"Latest batch: 2 done · 2 skipped" leading the tally.
