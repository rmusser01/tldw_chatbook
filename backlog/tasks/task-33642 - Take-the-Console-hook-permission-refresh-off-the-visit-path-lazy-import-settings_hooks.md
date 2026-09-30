---
id: TASK-33642
title: >-
  Take the Console hook-permission refresh off the visit path; lazy-import
  settings_hooks
status: To Do
assignee: []
created_date: '2026-09-30 15:00'
labels:
  - perf
  - console
  - hooks
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
#2922 (Console hooks) made every Console visit refresh hook-permission state from disk and made settings_screen import settings_hooks at module level. PERF-01's guards measured the cost when #2888 was rebased onto dev 75c06af39a: a visit went from 39 to 43 config admissions, 112 to 127-133 storage admissions and 35,351 to about 44,700 os.open calls, and the Settings pre-import pass from 556 to 557 modules. The owner approved re-pinning those ceilings on 2026-09-30 so the guards could land, on condition the cost is taken off the path afterwards. This task restores the previous ceilings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Console visit with unchanged hook configuration performs no hook-permission disk reads beyond the first visit; a changed hook file is still picked up on the next visit
- [ ] #2 settings_hooks is not imported by the pre-import pass (Settings route back to 45 added modules)
- [ ] #3 MAX_VISIT_STORAGE_UNITS and MAX_PASS_ADDED_MODULES are lowered back to 39/112/35,351 and 556 (or below), with the ADR-097 ledger updated
- [ ] #4 Hook review behaviour is unchanged: pending hooks still surface in the Console control bar and review modal
<!-- AC:END -->
