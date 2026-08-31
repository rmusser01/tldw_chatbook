---
id: TASK-25822
title: Disabled and low-emphasis actions render below readable contrast
status: Done
assignee: []
created_date: '2026-08-31 05:08'
updated_date: '2026-08-31 06:36'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A disabled primary button renders at roughly 1.4 to 1 against its background while the adjacent dismissive action renders near 14 to 1, so the dialog appears to offer only the dismissive choice. The same pattern makes the first-run final step's three exit controls render at roughly 1.65 to 1. A disabled control must still be legible enough to be understood as unavailable rather than absent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disabled control text meets at least 3 to 1 contrast against its background
- [ ] #2 A primary action remains the most visually prominent control in its dialog in every state
- [ ] #3 No interactive control renders below 3 to 1 in any state
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: Button:disabled stacked opacity:50% ON TOP of an already-dim $text-disabled, compositing to a measured 1.34:1 (test-reproduced, matching the 1.39:1 measured live). Textual's variant-nested &:disabled { text-opacity: 0.6 } also outranked a bare Button:disabled, so the rule needed .-style-default/.-style-flat specificity to win. Fix: new $ds-text-disabled-readable (#8a8a8a, 5.09:1 on the disabled surface) following the repo's measured-token convention; dropped the compounding opacity, keeping surface + weight as the non-colour cues. Deleted the now-redundant Button.model-import:disabled opt-out, which existed only to defeat the dimmer. Pinned by Tests/UI/test_task_25720_disabled_button_contrast.py across primary/default/model-import.
<!-- SECTION:NOTES:END -->
