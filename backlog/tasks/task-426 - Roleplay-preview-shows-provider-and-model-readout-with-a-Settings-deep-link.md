---
id: TASK-426
title: Roleplay preview shows provider and model readout with a Settings deep-link
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The preview pane gives no indication of which provider/model will answer (personas_preview_controller.py:270-276 hardwires character_defaults) and offers no way to inspect or change it. Console solves this with a Model section + Configure link; the preview should at least show the resolved provider/model and link to Settings > Providers & Models. Complements task-425.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preview pane displays the resolved provider and model that character replies will use
- [ ] #2 A visible affordance navigates to the provider configuration surface
- [ ] #3 Readout updates when the resolution changes (settings saved, config reloaded)
<!-- AC:END -->
