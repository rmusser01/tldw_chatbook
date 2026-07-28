---
id: TASK-1191
title: 'Retire _fit_height_to_content two-pass deferred settle'
status: To Do
assignee: []
created_date: '2026-07-27 21:30'
labels: [console, ui, layout]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleWorkspaceContextTray._fit_height_to_content settles over two deferred call_later passes plus a 0.01s timer on every tray state sync. Investigated twice during TASK-1142: not reproducible as a click-eating race under 15 rapid sync cycles, but it is real, separately-verifiable complexity that tests must work around, and a stale-geometry window in principle. Replace with a single-pass deterministic height computation (1142's estimator now covers the hard case) or document why two passes are load-bearing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tray height settles in one deterministic pass, or an ADR-style comment explains the two-pass necessity with a pinned test.
- [ ] #2 Existing tray/height tests pass without settle-window workarounds.
<!-- AC:END -->
