---
id: TASK-35569
title: Settings shows keyboard focus on the Inspector when F6 moves it there
status: To Do
created_date: 2026-10-18 22:50
dependencies: []
labels:
- model-config-redesign
- follow-up
- settings
- accessibility
priority: low
references:
- tldw_chatbook/css/features/_settings.tcss
- tldw_chatbook/UI/Screens/settings_screen.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's live capture phase (2026-10-18, capture 04e in qa/model-config-p9-captures/). On Providers & Models, two F6 presses move keyboard focus to the Inspector, but nothing on screen shows it. The rail and the detail pane each show their focused row with a █, while the Inspector looks the same focused or not. A third F6 puts the █ back on the rail. A keyboard user cannot tell where focus is, or that the arrow keys and the wheel now scroll the Inspector.

This predates Phase 9. Phase 9's CSS changes (111f8e48fc..6212bba292) touch no Inspector or focus rule, and no rule in css/features/_settings.tcss styles the Inspector's focus.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 211x44 and 235x52, when F6 moves focus to the Settings Inspector, a visible focus style marks it, and the style goes away when focus leaves
- [ ] #2 The style uses the existing focus tokens and does not change the Inspector's layout or width
- [ ] #3 A test under the real stylesheet fails before the fix: the focused Inspector paints the same as the unfocused one
<!-- AC:END -->
