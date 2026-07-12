---
id: TASK-185
title: >-
  Search/RAG polish batch: quiet no-sources gate, provider-test outcome toast,
  pluralization, moving-Run fix
status: In Progress
assignee: []
created_date: '2026-07-12 02:48'
updated_date: '2026-07-12 05:33'
labels:
  - ux
  - library
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11 findings across Search and adjacent surfaces: (1) the no-sources state stacks an 8-line Why/Next/Recovery/Owner dump plus three overlapping guidance lines with internal jargon ('Owner: Library source index') - regressing the quiet-gate principle; (2) Settings 'Provider test finished.' toast reports no pass/fail; (3) 'Matched conversation - 1 messages'; (4) the Run and Start ingest primary buttons shift 30-40px when gate helper lines collapse, breaking muscle memory; (5) keyword search misses plural forms (loops vs loop).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No-sources state shows one quiet gate line plus a single recovery action
- [ ] #2 Provider test reports success or failure with a reason
- [ ] #3 Result counts pluralize correctly
- [ ] #4 Primary action buttons keep a stable position across gate-state changes
- [ ] #5 Keyword search matches simple plural/singular variants or documents the limitation inline
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Quiet no-sources gate, outcome-bearing provider test toast, pluralization, and stable Run/Start-ingest button positions all fixed and live-verified on PR #604. Remaining AC: keyword-search plural/singular matching (or an inline limitation note) — untouched; live network reachability probe split to task-191.
<!-- SECTION:NOTES:END -->
