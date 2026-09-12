---
id: TASK-31698
title: Align transcript action and CSS expectations with current owners
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:39'
updated_date: '2026-09-05 18:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep transcript behavior and centering contracts accurate after failed-message action policy and stylesheet splitting changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failed assistant rows expose Retry without Continue at every reference width
- [x] #2 Empty-state CSS checks account for repeated selector declarations in generated sheets
- [x] #3 The full native transcript test file passes with geometry checks retained
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only alignment with established action policy and generated CSS cascade.
1. Reproduce five failures and inspect the action owner and repeated selector rules.
2. Update failed-row expectations and collect matching CSS declaration blocks.
3. Run the full transcript file and static checks without production changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned failed-message actions with the existing Console action owner: Retry remains available and Continue is prohibited. CSS assertions now collect exact base-selector declaration blocks across the generated cascade rather than accidentally reading the first scoped compact rule. No production changes or geometry relaxation. Original full-file RED: 5 failed/160 passed; final full file: 165 passed in 120.55s. Ruff lint and changed-region format checks passed. ADR not required (test-only current-contract repair); self-review complete.
<!-- SECTION:NOTES:END -->
