---
id: TASK-1478
title: >-
  Evals rail keeps creation affordances after first use (spec compliance)
status: Done
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
Found by live UAT (2026-07-30). All creation affordances in the Evals Catalog rail are empty-state-only: "Create sample bench" disappears once one bench exists, and "+ New dataset" / "Import…" disappear once one dataset exists (`library_rail.py` `_dataset_empty_actions`). After the first click the screen offers no creation path at all — a one-way trapdoor to read-only, verified live in two profiles.

This violates the approved design spec (2026-07-25-evals-console-rebuild-design.md, "Screen IA and layout"): "Each of `Benches` and `Datasets` carries its own creation affordance in the section header — a new bench and a new snippet set are reachable without first finding an empty state."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] With one or more benches present, a bench-creation affordance (the sample bench, until task-1482 lands authoring) remains reachable from the Benches section
- [x] With one or more datasets present, "+ New dataset" and "Import…" remain reachable from the Datasets section
- [x] First-run empty states keep their current guidance copy
- [x] Tests assert the affordances exist in non-empty rail states
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move rail creation affordances from empty-state-only to always-rendered at the top of each section body
2. Keep ids, gate logic, and empty-state hint copy unchanged
3. Tests for non-empty presence and collapsed-section absence
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commit c41d679c9. `_dataset_empty_actions` became `_dataset_actions` and renders unconditionally at the top of the Datasets body; the sample-bench button renders in the Benches body whenever the provider gate passes, benches present or not. Widget ids unchanged so on_button_pressed wiring is untouched; affordances collapse with their section. The reviewer traced the full render matrix: no duplicate "Open Settings" is structurally possible, and the one zero-affordance cell (benches exist + gate fails) is unreachable today because no delete_model path exists — re-check that cell when task-1482 adds a non-sample bench-creation path. Verified live: affordances present alongside data in two profiles.
<!-- SECTION:NOTES:END -->
