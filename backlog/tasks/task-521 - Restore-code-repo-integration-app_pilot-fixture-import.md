---
id: TASK-521
title: Restore code-repo integration app_pilot fixture import
status: Done
assignee: []
created_date: '2026-07-24 18:44'
updated_date: '2026-07-24 19:08'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the code-repository integration harness after automated unused-import cleanup removed the pytest fixture import used by every app-pilot test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The integration module registers the shared app_pilot fixture explicitly
- [x] #2 All code-repo workflow tests can enter the Textual app harness
- [x] #3 The fixture import is documented as intentional and Ruff-clean
- [x] #4 No production code changes
- [x] #5 The full code-repo integration file passes
- [x] #6 Task documentation records the merge-base failure, regression commit, ADR decision, verification, and implementation notes
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the shared app_pilot import in Tests/integration/test_code_repo_integration.py so pytest registers the fixture used by all six workflow tests. Added an explicit pytest-registration comment, F401 annotation, and file-level F811 suppression because the fixture parameter names intentionally shadow the imported registration symbol.

The merge base fails collection/setup because app_pilot is absent. Commit f56eb4121 (style(ruff): remove unused imports (F401)) introduced the regression by deleting the fixture-only import. The restoration itself is commit cf7b271d8. After TASK-522 isolated app database paths, the full integration module passes: 6 passed. Ruff, format, and diff checks passed, and independent review approved the narrowly scoped test-only change.

ADR required: no. This restores existing pytest fixture registration without changing production code, dependencies, or architecture.
<!-- SECTION:NOTES:END -->
