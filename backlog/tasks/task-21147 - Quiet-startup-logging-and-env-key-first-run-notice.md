---
id: TASK-21147
title: Quiet startup logging and env-key first-run notice
status: To Do
assignee: []
created_date: '2026-08-25 06:15'
labels:
  - ux
  - app
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings G-7, E-1 (findings.md): every cold start prints a wall of DEBUG/WARNING log lines (including 'CRITICAL DEBUG:') to the terminal before the TUI mounts; with a provider env var set, a fresh install boots straight to Console with no acknowledgement and no pointer to the setup wizard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A default cold start prints no DEBUG/INFO log lines to the terminal before the TUI mounts (WARNING+ only); verbose logging remains available via config or env var
- [ ] #2 First run with a provider env key shows a one-time dismissible notice naming the detected key and how to run setup
- [ ] #3 The notice never reappears after dismissal
<!-- AC:END -->
