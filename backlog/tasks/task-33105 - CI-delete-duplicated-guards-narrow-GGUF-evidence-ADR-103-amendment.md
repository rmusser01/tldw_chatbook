---
id: TASK-33105
title: 'CI: delete duplicated guards, narrow GGUF evidence, ADR-103 amendment'
status: To Do
assignee: []
created_date: '2026-09-27 20:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec 2026-09-27-ci-conflicts-and-waste-design.md parts C1, C2 and E.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 css-bundle-guard and backlog-guard deleted; bundle and backlog-id checks pinned to run on dev/main pushes
- [ ] #2 GGUF evidence paths narrowed to GGUF code and pinned in Tests/CI
- [ ] #3 GGUF UI test files added to the UI census only if green on the minimal dependency set
- [ ] #4 ADR-103 amended for the nightly cadence change and the removed guards
<!-- AC:END -->
