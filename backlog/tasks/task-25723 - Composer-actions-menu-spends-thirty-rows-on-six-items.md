---
id: TASK-25723
title: Composer actions menu spends thirty rows on six items
status: Done
assignee: []
created_date: '2026-08-31 05:09'
updated_date: '2026-08-31 13:52'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The composer menu spreads six actions over roughly thirty rows with items centred rather than left aligned, anchors itself at the top left while its trigger sits at the bottom of the screen, and shows no keyboard accelerators. Reasons for disabled items are present and well written but separated from their action by blank rows, weakening the association. The result occludes the transcript to present very little.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Menu items are left aligned and vertically compact enough to scan without scrolling
- [ ] #2 The menu is anchored adjacent to the control that opens it
- [ ] #3 A disabled item's reason is visually bound to that item
- [ ] #4 Keyboard accelerators are shown for actions that have them
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rows were height:3 plus a trailing margin-bottom:1 -- four rows each, ~30 for six actions -- with every label centred. Set text-align:left and margin:0, which also pulls each disabled row's explanatory reason back against the row it explains (the trailing gap was what broke that association).

KEY CONSTRAINT, learned from the sibling contract test: the rule had to go in the APP stylesheet (_agentic_terminal.tcss), NOT the modal's DEFAULT_CSS. test_master_shell_design_system_contract records that Button's own rules outrank a modal's DEFAULT_CSS and that 'an identical rule inside the modal measured no change' -- the existing :disabled rule already lives there for exactly this reason. Left height:3 in DEFAULT_CSS and added a comment naming the split so the next reader does not re-litigate it.

NOT addressed, and honestly out of scope for a CSS pass: the menu anchors top-left while its trigger is at the bottom of the screen, and shows no accelerators. Both need the modal's positioning and an accelerator model, not styling.

Baseline confirmed unchanged: test_impersonate_payload_obeys_the_provider_contract and test_transcript_trimming_keeps_newest_and_stays_user_first fail on clean dev.
<!-- SECTION:NOTES:END -->
