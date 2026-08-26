---
id: TASK-16253
title: Reconcile shape-specific focus contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 13:34'
updated_date: '2026-08-14 13:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the focus accessibility suite by distinguishing the shared Input/TextArea border treatment from Select's established shape-specific focus geometry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Input and TextArea tests prove the shared thin-border and bottom-emphasis pattern.
- [x] #2 Select tests prove the global parent rule remains color-only while compact-select geometry retains a visible focus cue.
- [x] #3 The focus accessibility suite and CSS bundle checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a test-only reconciliation with established focus styling; no architecture boundary changes.

1. Preserve the stale Select border assertion as RED evidence and verify the CSS source history.
2. Split the test contract between Input/TextArea shared focus borders and Select shape-specific focus behavior.
3. Run focused accessibility, CSS bundle, lint, and diff-hygiene checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Split the stale shared-focus assertion so Input and TextArea retain their common thin-border contract while Select preserves its intentional shape-specific geometry.
- Added explicit evidence that the global Select rule remains border-free and that compact SelectCurrent receives its established focus background.
- Verified the focused accessibility module (9 passed), CSS bundle/consolidation gates, Ruff, formatting, diff hygiene, and the full 25-file checkpoint (221 passed, 1 skipped, 2 existing warnings).
- ADR check: no ADR required; this reconciles tests with established styling and changes no production behavior.
<!-- SECTION:NOTES:END -->
