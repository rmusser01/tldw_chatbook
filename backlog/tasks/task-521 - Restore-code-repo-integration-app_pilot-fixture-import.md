---
id: TASK-521
title: Restore code-repo integration app_pilot fixture import
status: In Progress
assignee: []
created_date: '2026-07-24 18:44'
updated_date: '2026-07-24 18:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the code-repository integration harness after automated unused-import cleanup removed the pytest fixture import used by every app-pilot test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The integration module registers the shared app_pilot fixture explicitly
- [ ] #2 All code-repo workflow tests can enter the Textual app harness
- [ ] #3 The fixture import is documented as intentional and Ruff-clean
- [ ] #4 No production code changes
- [ ] #5 The full code-repo integration file passes
- [ ] #6 Task documentation records the merge-base failure, regression commit, ADR decision, verification, and implementation notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the missing-fixture error on feature branch and merge base and trace it to automated F401 cleanup.
2. Restore the shared app_pilot fixture import with an explicit lint annotation explaining pytest registration.
3. Run the full code-repo integration file.
4. Run Ruff format/check and git diff --check; independently review before completion.

ADR required: no
ADR path: N/A
Reason: This restores an existing test harness fixture registration removed by mechanical lint cleanup; it changes no production code, dependency, or architecture.
<!-- SECTION:PLAN:END -->
