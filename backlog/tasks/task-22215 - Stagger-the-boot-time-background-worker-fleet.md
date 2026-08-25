---
id: TASK-22215
title: >-
  Stagger the boot-time background worker fleet
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - startup
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22215).

Boot-time concurrent workers went 4 -> 7 since the pin (new: chachanotes-fts-backfill,
the initial-screen pre-import thread, actor-pack recovery relocation). Under the GIL these
CPU-bound import/tokenize threads plus the Textual pump share one interpreter during the
first seconds after mount — worst on the first post-upgrade boot when 22200's backfill
runs to completion alongside the pre-importer. Each worker is individually justified; the
aggregate is what the user feels.

## Acceptance Criteria

- [ ] Boot workers are census'd (a test pins the set, so an eighth is a reviewed decision) and started with an explicit priority/stagger policy
- [ ] Input latency during the first 5 s after mount measured before/after on a warm boot and on a simulated first-post-upgrade boot
- [ ] Backfills yield to foreground work (coordinate with TASK-22200)
