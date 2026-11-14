---
id: TASK-40543
title: 'Focused Input selected text is under 4.5:1 on most themes'
status: To Do
assignee: []
created_date: '2026-11-12 09:00'
labels:
  - follow-up
  - contrast-fix
dependencies: []
references:
  - tldw_chatbook/css/Themes/themes.py
  - Tests/UI/test_theme_contrast.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The text selected in a focused Input (`$input-selection-background` and `-foreground`, which is what a keyboard return to a field shows) is under 4.5:1 as painted on most themes: camono is 2.49:1, litestep_dark 3.39:1, agentic_terminal 4.01:1 on `$panel`. It exists on dev today (`4b90eb860b`). No open PR fixes it: PR #3468 (TASK-38229) is about the caret. The Workflows search fixed its own field (TASK-33003.15 round 5); nothing fixes the general Input. Found incidentally by the Roleplay B2b follow-up burn-down (2026-11-07); evidence in `fu-caret-report.md` finding 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selected text in a focused Input reads at 4.5:1 or better on every measurable theme over each of the three panes, measured as painted
- [ ] #2 Themes that already read keep their painted selection
- [ ] #3 A gate covers all measurable themes and is red on dev for the failing rows
- [ ] #4 No boot-parsed CSS growth
<!-- AC:END -->
