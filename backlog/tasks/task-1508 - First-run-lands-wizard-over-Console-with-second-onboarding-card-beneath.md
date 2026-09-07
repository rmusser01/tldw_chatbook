---
id: TASK-1508
title: First-run lands wizard over Console with second onboarding card beneath
status: Done
assignee: []
created_date: '2026-07-31 00:23'
updated_date: '2026-07-31 02:05'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
_resolve_initial_shell_route also routes Home when should_offer_wizard() is true — the ephemeral _first_run flag is lost to config force-reloads on real installs, which was landing the wizard over Console's own Get-started card. Covered by the existing app-level pin test (wizard over Home).
<!-- SECTION:NOTES:END -->
