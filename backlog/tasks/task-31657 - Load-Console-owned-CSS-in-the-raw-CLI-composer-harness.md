---
id: TASK-31657
title: Load Console-owned CSS in the raw CLI composer harness
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:29'
updated_date: '2026-09-05 17:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore meaningful raw CLI danger-style and collapsed-geometry verification after the Console stylesheet split without changing production styling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing raw CLI danger-color and collapsed-geometry assertions pass with the stylesheet loaded by the real Console.
- [x] #2 The full command-composer test file passes and scoped static checks remain clean.
- [x] #3 Collapsed raw CLI danger presentation follows the active theme semantic error color in both light and dark modes rather than a retired fixed RGB value.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce unchanged raw CLI danger-color failure and compare widget-only harness CSS sources with production ChatScreen ownership. 2. Add existing Console CSS to the harness. A second RED reveals the retired fixed RGB expectation: verify TASK-31264 and use the independent active-theme text-error variable for expected color, exercising light and dark modes while retaining exact semantic color, focus and geometry assertions. 3. Run the focused regression, theme-contrast suite and full command-composer file; compare scoped static diagnostics to baseline. ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md (existing). Reason: test-only alignment with existing screen-owned CSS and approved theme-aware error token; no runtime or styling changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The widget-only raw CLI harness now includes ChatScreen.CSS_PATH, matching production Console-owned styling. Existing RED then exposed a second obsolete assumption: TASK-31264 replaced fixed #ff8fa3 with polarity-aware text-error. The test independently resolves that theme variable and retains exact color, class, bold, background, geometry, focus and draft-restoration checks in both dark and light modes. Removed one pre-existing unused import in the touched test file. No production changes or new ADR; existing ADR-097 and TASK-31264 apply. Evidence: original color regression RED; 144 focused composer/theme checks passed; final full command-composer file 104 passed in 238.98s. Full-file Ruff lint and changed-region formatting pass, git diff check clean; unrelated legacy full-file formatting preserved. Self-review found no relaxed assertion or behavior change.
<!-- SECTION:NOTES:END -->
