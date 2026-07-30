---
id: TASK-1478
title: >-
  Evals rail keeps creation affordances after first use (spec compliance)
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
Found by live UAT (2026-07-30). All creation affordances in the Evals Catalog rail are empty-state-only: "Create sample bench" disappears once one bench exists, and "+ New dataset" / "Import…" disappear once one dataset exists (`library_rail.py` `_dataset_empty_actions`). After the first click the screen offers no creation path at all — a one-way trapdoor to read-only, verified live in two profiles.

This violates the approved design spec (2026-07-25-evals-console-rebuild-design.md, "Screen IA and layout"): "Each of `Benches` and `Datasets` carries its own creation affordance in the section header — a new bench and a new snippet set are reachable without first finding an empty state."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] With one or more benches present, a bench-creation affordance (the sample bench, until task-1482 lands authoring) remains reachable from the Benches section
- [ ] With one or more datasets present, "+ New dataset" and "Import…" remain reachable from the Datasets section
- [ ] First-run empty states keep their current guidance copy
- [ ] Tests assert the affordances exist in non-empty rail states
<!-- AC:END -->
