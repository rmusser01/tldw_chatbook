---
id: TASK-1484
title: >-
  Rename affordances for Evals benches and datasets
status: To Do
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from live UAT (2026-07-30). Datasets are created as "Untitled dataset <hex>" and benches as "<name> <hex>"; nothing in the UI can rename either, so the catalog fills with hex-suffixed placeholders. Detail panes are read-only Statics by design in the shipped slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A dataset can be renamed from its detail pane
- [ ] A bench can be renamed from its detail pane
- [ ] Rail rows and grid headers reflect the new name immediately
<!-- AC:END -->
