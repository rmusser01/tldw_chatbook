---
id: TASK-33030
title: 'CI: derive diagnostic-inventory totals from rows (schema 4)'
status: To Do
assignee: []
created_date: '2026-09-27 17:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stop committing the inventory summary totals that conflicted on 102 of 127 two-sided sync merges; spec 2026-09-27-ci-conflicts-and-waste-design.md part A.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 build_inventory emits schema 4 with no summary; totals derived by inventory_summary()
- [ ] #2 All five consumer test files keep the dev baseline red set exactly
- [ ] #3 Committed inventory regenerated; only schema_version and summary lines changed
- [ ] #4 ADR-029 amendment recorded
<!-- AC:END -->
