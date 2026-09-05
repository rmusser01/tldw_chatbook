---
id: TASK-31552
title: Console Assistant-turn harness omits the split production stylesheet
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 00:49'
updated_date: '2026-09-05 01:24'
labels:
  - console
  - tests
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Console Assistant-turn presentation tests to the real production stylesheet stack after Console-owned CSS moved out of the monolithic bundle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Assistant activity labels and statuses retain their bounded production geometry.
- [x] #2 Roleplay and failure semantic fills remain distinct in both shipped base themes.
- [x] #3 All Assistant-turn compositor contrast cases paint readable foreground and background colors.
- [x] #4 The complete focused Assistant-turn test module passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the Assistant-turn file failures and compare its claimed production CSS stack with TldwCli.CSS_PATH.
2. Add the generated Console-owned split stylesheet to the styled harnesses without changing production CSS.
3. Run the complete Assistant-turn module, focused CSS-integrity coverage, Ruff, and git diff checks.
4. Record evidence and complete the task.

ADR required: no
ADR path: N/A
Reason: the production Console styling contract is already established by TASK-25812 and TASK-19426; this only repairs a stale test harness.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the generated Console stylesheet to both styled Assistant-turn harnesses so their geometry and compositor assertions exercise the same base-plus-owner stack as production.
- No production CSS changed.
- Evidence: the complete module passes 39/39; Ruff and diff checks pass.
- ADR required: no; TASK-25812 already owns the stylesheet split boundary.
<!-- SECTION:NOTES:END -->
