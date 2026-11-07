---
id: TASK-38229
title: 'Textual Input caret is under 3:1 on some themes'
status: To Do
assignee: []
created_date: '2026-10-30 18:43'
labels:
  - a11y
  - css
dependencies: []
references:
  - Tests/UI/test_theme_contrast.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual's own caret colours over a letter measured 2.53:1 on pastel_dreams (Roleplay frame B2b's contrast work). It exists on dev today for every Input; the Workflows search restyles its caret, and B2b does not fix the others. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Input's caret clears 3:1 against its fill on every measurable theme, pinned by a painted test
- [ ] #2 No Input's typed text drops below 4.5:1 as a result
<!-- AC:END -->
