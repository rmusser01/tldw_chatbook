---
id: TASK-39178
title: Keep first-run Console setup actions visible at 80 by 24
status: Done
assignee: []
created_date: 2026-11-11 11:03
labels:
- uat
- console
- onboarding
- accessibility
dependencies: []
priority: medium
updated_date: 2026-11-11 20:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console's first-run **Get started** card lets keyboard focus move below the viewport at a common terminal size. The setup wizard also clips the small-terminal guidance at the same size.

### Reproduction

1. Launch local `origin/dev` `6bd97b112f` in a native 80-column by 24-row Windows terminal with a fresh profile and no provider credentials.
2. Note that the wizard's bottom line ends at **Everything still**.
3. Exit setup before configuring a provider. Wait for Console's **Get started** card to appear.
4. Press **Tab** from **Set up provider**. The focus border moves to the next button but the terminal footer covers its label. The card does not scroll to reveal it.

### Evidence

- [First-run wizard at 80×24](../../qa/uat-core-workflows-2026-11-11/native-80x24-wizard.txt)
- [Console gate before Tab](../../qa/uat-core-workflows-2026-11-11/native-80x24-provider-gate.txt)
- [Focused button below the viewport](../../qa/uat-core-workflows-2026-11-11/native-80x24-provider-gate-tab.txt)

TASK-441 addressed the width of step text. This reproduction concerns the vertical space for the card's actions and the wizard's guidance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 At 80×24, Tab and Shift+Tab keep the focused Get started action and its full label visible.
- [x] #2 The secondary Write a note in Library action can be found and activated using only the keyboard.
- [x] #3 At 80×24, the wizard's small-terminal guidance is fully readable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: No
ADR path: N/A
Reason: Use the existing layout and focus rules at compact terminal sizes.

Pin the painted focused action and wizard guidance at 80×24, then size the existing card and guidance within the viewport and verify keyboard activation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The existing Console setup card has a bounded height and scrolls focused actions into view. The wizard's small-terminal hint wraps within the viewport. Textual Pilot at 80×24 verified painted action labels, Tab in both directions, keyboard activation of Write a note in Library, and the full wizard guidance. The combined onboarding and token-governance run passed 15 tests. The CSS bundle was rebuilt and lint passed. Existing ADR-150 and ADR-161 apply.
<!-- SECTION:NOTES:END -->
