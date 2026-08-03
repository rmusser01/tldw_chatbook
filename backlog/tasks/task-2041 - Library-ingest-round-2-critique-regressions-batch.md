---
id: TASK-2041
title: >-
  Library ingest round-2 critique regressions batch (focus labels, toast counts, worker logging, Clear resurrection)
status: In Progress
assignee: []
created_date: '2026-08-03 02:00'
labels:
  - library
  - ingest
  - ux
  - regression
priority: high
dependencies: []
---

## Description (the why)

The round-2 dual-agent critique (snapshot `2026-08-03T01-33-07Z…`, 24/40)
caught two regressions introduced by this arc's own fixes plus one residual
and one new defect, all small and well-understood:

1. task-2014's `outline: heavy $accent` on 1-row compact buttons draws over
   the label row — every focused action button renders as a label-less
   heavy-border box, and the two-press "Press again to clear N finished"
   confirm label is invisible exactly when it must be read.
2. task-2015's settle toast computes `imported = done_delta`, so dedup
   matches report as "imported" ("Ingest finished — 1 imported" directly
   above a row saying "nothing new was imported").
3. The task-2016 worker-noise guard covered loguru + the `warnings` module
   but not stdlib logging's auto-basicConfig fallback:
   `WARNING:root:OpenTelemetry not installed…` still paints over the chrome
   on a fresh process's first submit.
4. New: after pressing the path "Clear" button, a focus click into the
   empty field re-materializes the previous text (reproduced twice,
   deliberately isolated; concatenated garbage paths result).

## Acceptance Criteria (the what)

- [ ] A focused compact button on the ingest canvas shows its label,
      readably, with a monochrome-visible focus indicator (incl. the armed
      "Press again to clear N finished" label).
- [ ] The settle toast distinguishes imported from already-in-Library
      matches (a dedup-only batch never says "imported").
- [ ] A fresh process's first submit paints no stdlib-logging output over
      the TUI (root logger silenced in spawn workers alongside loguru and
      warnings).
- [ ] After pressing Clear, the path field stays empty through focus
      clicks and recomposes.
