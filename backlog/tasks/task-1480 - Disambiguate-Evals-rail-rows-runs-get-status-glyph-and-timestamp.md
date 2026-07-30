---
id: TASK-1480
title: >-
  Disambiguate Evals rail rows: runs get status glyph and timestamp
status: To Do
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by live UAT (2026-07-30). After one sample bench, the rail shows three near-identical rows: bench `loaded-nouns (sample) fbf8b2d0`, dataset `loaded-nouns (sample) 3a7644f6`, run `loaded-nouns (sample) fbf8b2d0` — the bench and run rows are byte-identical. The blocked-reason copy relies on the user distinguishing bench rows from run rows, and nothing visually distinguishes them; hex suffixes are the only difference and they carry no meaning.

The design spec's rail mock renders run rows as `● 14:31 run` / `✓ 14:02 run` / `✗ 13:55 run` — status glyph plus start time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Run rows render a textual status marker plus start time (spec mock shape), not a bare copy of the bench name; status is never conveyed by color alone
- [ ] Bench, dataset, and run rows are distinguishable at a glance without reading hex suffixes
- [ ] Tests assert the run-row label format for completed, failed, and cancelled runs
<!-- AC:END -->
