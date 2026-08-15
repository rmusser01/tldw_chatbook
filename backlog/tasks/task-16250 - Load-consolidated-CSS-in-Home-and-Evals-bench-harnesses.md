---
id: TASK-16250
title: Load consolidated CSS in Home and Evals bench harnesses
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 12:55'
updated_date: '2026-08-14 12:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the remaining Home and Evals bench integration harnesses load the same consolidated widget CSS sources as production after TASK-15450.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home navigation and content geometry use production-equivalent widget defaults.
- [x] #2 Evals bench controls are visible and pointer-operable at the tested viewports.
- [x] #3 Full Home and Evals bench modules pass or any remaining failures are independently classified.
- [x] #4 Static and task hygiene checks are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a test-harness correction using the existing consolidated-CSS helper.

1. Preserve representative unstyled geometry and off-screen click failures as RED evidence.
2. Change only HomeHarness and EvalsHarness from bare App to ConsolidatedCSSApp.
3. Run representative nodes and the full affected modules.
4. Re-run checkpoint chunk 53 or isolate any unrelated remaining destination failures.
5. Run static checks, self-review, document evidence, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Migrated the two remaining bare `App` harnesses, `HomeHarness` and `EvalsHarness`, to the existing `ConsolidatedCSSApp` helper so widget-default self/scoped sheets load alongside their existing production bundle.
- RED evidence showed a 19-row navigation bar and off-screen BenchEditor controls. The representative Home geometry nodes and Evals Save node turned green immediately after the two inheritance changes.
- Full verification passed 60 Home tests and 80 Evals bench-editor tests. The checkpoint's compact Home case also passed on a fresh last-failed rerun.
- Six remaining destination-parity failures are independent current geometry/copy contracts: four Scheduling thresholds, one MCP inspector viewport assertion, and one Settings marker-class assertion. They are left for the next atomic task.
- Scoped Ruff lint and `git diff --check` passed; `test_home_screen.py` is formatted. `test_evals_bench_editor.py` retains its exact HEAD formatter drift and the changed lines add no new format delta.
- ADR check: no ADR was required because the task uses the repository's existing production-equivalent CSS harness helper without changing runtime behavior.
<!-- SECTION:NOTES:END -->
