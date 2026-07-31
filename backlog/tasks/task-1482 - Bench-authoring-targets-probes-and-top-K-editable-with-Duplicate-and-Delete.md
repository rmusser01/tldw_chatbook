---
id: TASK-1482
title: >-
  Bench authoring: targets, probes, and top-K editable with Duplicate and Delete
status: In Progress
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from live UAT (2026-07-30). Three of the five results-grid lenses can never show real data through the shipped UI: the Probe lens reports "no probes configured" and there is no probe authoring; the Δ baseline lens and spread sort need two or more targets and there is no target picker. The only reachable bench is the hardwired single-target sample, so the analysis engine's cross-target features are complete but unreachable. Imported datasets are equally stranded: the blocked copy says "select a bench that uses this dataset instead", but no bench can ever be pointed at one.

The design spec already covers this (bench editor mock with `[ + Add target ]`, probes row, and `[ Duplicate ] [ Delete ]` inspector actions); it was deferred out of the vertical slice. This is the largest remaining gap between the shipped screen and the approved design. Needs its own plan; not part of the 1476-1481 fix batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A bench's targets can be added and removed from the bench editor (target picker over configured models)
- [ ] Probes and top-K are editable on a bench
- [ ] A bench can be created against any existing dataset (closing the imported-dataset dead end)
- [ ] Duplicate and Delete exist for benches per the spec's inspector actions
- [ ] The Probe and Δ baseline lenses are reachable with real data through UI-authored benches
<!-- AC:END -->
