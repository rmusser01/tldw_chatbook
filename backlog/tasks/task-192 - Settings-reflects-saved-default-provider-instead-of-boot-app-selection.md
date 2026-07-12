---
id: TASK-192
title: Settings reflects saved default provider instead of boot app selection
status: To Do
assignee: []
created_date: '2026-07-12 12:44'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual from the 2026-07 core-loop waves (same family as the fixed task-177): after the setup card's one-click local connect writes chat_defaults.provider, the Settings Providers & Models category still displays the boot-time app selection (e.g. OpenAI, 'Provider source: Current app selection') until the user reselects, while Console correctly runs the saved provider. Evidence: Docs/superpowers/qa/core-loop-upgrades-2026-07/README.md residuals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Providers & Models after a config-level provider change shows the saved default provider without manual reselection,Readiness/test rows reflect the same provider Console resolves
<!-- AC:END -->
