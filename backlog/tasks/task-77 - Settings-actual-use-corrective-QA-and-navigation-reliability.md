---
id: TASK-77
title: Settings actual-use corrective QA and navigation reliability
status: In Progress
labels:
- settings
- ux
- qa
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the Settings configuration hub through actual rendered app use and correct any confirmed usability blockers in category navigation, field editing, dropdown selection, save/revert/test feedback, and keyboard operation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Actual rendered Settings walkthrough verifies category navigation works with mouse and keyboard.
- [ ] #2 Provider, model, endpoint, credential, numeric, and toggle controls remain readable while focused and while text is entered.
- [ ] #3 Dropdown selection works without clipping, hidden selection state, or invalid blank provider persistence.
- [ ] #4 Save, revert, validation, and test actions show clear status/recovery feedback and do not block the UI for long work.
- [ ] #5 Any confirmed blocker is covered by a failing regression before the production fix.
- [ ] #6 Final CDP screenshot evidence is captured and explicitly approved before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the Settings screen in textual-web from an isolated config/data home so actual behavior is observable without contaminating local user state.
2. Use CDP/browser automation to verify rendered category navigation, provider dropdown selection, input focus visibility, save/revert/test feedback, and keyboard traversal.
3. Record any confirmed blocker with exact reproduction steps and root-cause evidence before editing production code.
4. Add a focused failing regression for the first confirmed blocker, then implement the smallest safe fix.
5. Rerun focused Settings tests, run diff hygiene, capture actual final screenshots, and wait for user approval before PR creation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Slice 1 addressed the confirmed endpoint-field blocker found during CDP QA: clicking a URL-valued provider endpoint in textual-web opened the browser URL instead of behaving like an editable Settings input. Added a URL-safe endpoint input renderer that breaks browser autolinking in display text while preserving the raw value for validation and saving. Added focused regressions for the display transform and endpoint widget composition, then verified with Settings tests and CDP screenshot evidence approved by the user.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
