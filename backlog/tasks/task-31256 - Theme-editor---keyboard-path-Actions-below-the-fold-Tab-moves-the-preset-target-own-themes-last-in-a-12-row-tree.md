---
id: TASK-31256
title: 'Theme editor - keyboard path: Actions below the fold, Tab moves the preset
  target, own themes last in a 12-row tree'
status: To Do
created_date: 2026-09-04 05:24
dependencies:
- TASK-31254
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply/Save/Reset/Generate sit after 10 colour rows and 40 preset swatches: 50 Tabs from Primary, below the fold on a 52-row terminal while the State banner says 'buttons below'. Tabbing through the colour inputs to reach a swatch updates last_focused_color_input, so a keyboard user's preset always lands on Error (live: Blues-0 filled Error). The user's own themes are the last leaves after 58 shipped ones in a 12-row Tree with no Home/End key; the root is collapsed with ten blank rows. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Apply, Save, Reset and Generate are reachable without passing through the preset grid (Actions row rendered before Color Presets, or pinned above the palette)
- [ ] #2 Focusing a preset swatch does not change which colour it will fill; the target is the last colour input the user edited or focused by choice, default Primary, and is named in visible text
- [ ] #3 User Themes is the first group in the tree and the shipped group starts collapsed; the tree opens expanded to the user's themes instead of a collapsed root
- [ ] #4 The tree hint copy still mentions New and theme (pinned) and now describes the new order
- [ ] #5 Tests: Tab order test (Actions before presets), preset-target-unchanged-by-focus test, tree order test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
