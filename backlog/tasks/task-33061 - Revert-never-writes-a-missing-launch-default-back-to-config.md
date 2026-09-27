---
id: TASK-33061
title: Revert never writes a missing launch default back to config
status: To Do
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. revert_theme persists previous_launch_default unconditionally, so reverting after a launch-default-missing start rewrites the broken name to config and the 'Launch default missing' notice returns. The Revert chip also lingers when it would change nothing. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Revert leaves the saved launch default untouched when the previous one is not an available theme, and its label says so
- [ ] #2 The Revert chip is hidden when reverting would change neither the active theme nor the launch default
- [ ] #3 Existing Try/Use/Revert tests are updated to the new contract and pass
<!-- AC:END -->
