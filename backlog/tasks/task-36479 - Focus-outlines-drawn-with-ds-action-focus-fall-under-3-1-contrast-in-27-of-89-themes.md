---
id: TASK-36479
title: 'Focus outlines drawn with $ds-action-focus fall under 3:1 contrast in 27 of 89 themes'
status: To Do
assignee: []
created_date: '2026-10-23 11:02'
labels:
  - ui
  - accessibility
dependencies: []
references:
  - Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exists on dev today and is not fixed by Roleplay frame B2a (TASK-33910.3): the focus edge token $ds-action-focus measures under 3:1 against $surface in 27 of 89 themes, and dev draws focus outlines with it on 23 stylesheets today, among them Library's and Settings'. B2a's own items list already uses $ds-active-fg for its focus edge (owner decision 2026-10-23, "Both (Rec.)": file this follow-up and amend the Roleplay spec to name $ds-active-fg). Fix direction: $ds-active-fg (the Settings rail precedent) or a re-tuned token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every focus edge drawn on dev clears 3:1 against its surface on every measurable theme
- [ ] #2 A contrast test pins it, so a token change that drops a theme under 3:1 fails
<!-- AC:END -->
