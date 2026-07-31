---
id: TASK-1508
title: First-run lands wizard over Console with second onboarding card beneath
status: To Do
assignee: []
created_date: '2026-07-31 00:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: _first_run in-memory flag is lost to a config force-reload before routing, so the wizard opens over Console (whose Get-started card lurks beneath) instead of Home — Esc reveals a second onboarding surface. Pre-existing routing quirk, disorienting with the wizard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh-config boot routes the initial screen to Home beneath the wizard
- [ ] #2 No double-onboarding surface visible after Esc on a fresh install
<!-- AC:END -->
