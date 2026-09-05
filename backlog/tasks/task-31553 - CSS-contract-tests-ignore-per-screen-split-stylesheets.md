---
id: TASK-31553
title: CSS contract tests ignore per-screen split stylesheets
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 00:52'
updated_date: '2026-09-05 01:24'
labels:
  - tests
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make app-style selector contracts and focused render harnesses inspect the generated Console, Library, and Settings sheets that production loads after TASK-25812.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Global composed-class coverage inspects the complete app stylesheet set.
- [x] #2 Console and Library selector contracts accept rules in their owning generated split sheets.
- [x] #3 Speech and Settings render harnesses load their owning split sheet.
- [x] #4 Focused split-sheet contract tests pass without changing production CSS.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replay the red selector-contract nodes and compare each test source with the TASK-25812 generated split outputs.
2. Replace bundle-only app-style reads with the shared app_css_text union and add the appropriate owner sheet to direct render harnesses.
3. Run the focused red set, each touched contract module where practical, the CSS sync check, Ruff, and git diff checks.
4. Record evidence and complete the task.

ADR required: no
ADR path: N/A
Reason: TASK-25812 already defines CSS ownership and load order; this aligns stale tests with that accepted boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced monolithic-bundle reads with the shared complete app stylesheet union and loaded owner sheets in direct Console, Library, and Settings render harnesses.
- Updated generated-widget CSS ownership assertions and documented the small set of deliberately unstyled semantic classes; removed one stale exemption now proven styled by the Console sheet.
- Evidence: 22 focused selector contracts pass, and nine sampled CI compositor/geometry regressions now pass. Two sampled Console behavior assertions remain unrelated stale contracts (a one-row header placement expectation and a removed tab-strip class) and were not folded into this CSS task. The broader 535-test Library/CSS diagnostic completed with 521 passes and 14 independently classified residual failures.
- Ruff and diff checks pass; production CSS was not changed.
- ADR required: no; TASK-25812 already defines stylesheet ownership and load order.
<!-- SECTION:NOTES:END -->
