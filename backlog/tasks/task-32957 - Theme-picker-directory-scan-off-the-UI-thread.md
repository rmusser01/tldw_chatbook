---
id: TASK-32957
title: Theme picker directory scan off the UI thread
status: To Do
assignee: []
created_date: '2026-09-25 21:00'
labels:
  - settings
  - theme
  - perf
priority: medium
dependencies:
  - TASK-32948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings ▸ Theme lists saved theme files through a backup-scoped scan that runs on the UI thread. With 50 theme files it measured a 254 ms median (240–287 ms over 5 runs; 66 ms with 10 files), about 5–6.6 ms per file, almost all of it in the backup layer's per-file scope. TASK-32948's Qodo review fixes stopped theme switches (Use, Try, Revert, the command palette) from rescanning (Qodo 4107495860). The scans that remain run when the Theme page mounts, on Back from the editor, and after each file action (Save, Save as, Rename, Delete, Import, Export). These still freeze the UI for longer than the 100 ms at which CLAUDE.md's performance rules call for a worker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Settings ▸ Theme, pressing Back from the editor, and finishing a file action never block the UI thread for more than 100 ms with 50 saved theme files
- [ ] #2 While a scan is in flight, the picker shows the last good listing (or a loading state on first open) and stays responsive to keys
- [ ] #3 A scan result that arrives after a newer scan started, or after a file action, never overwrites the newer listing
- [ ] #4 A backup/recovery pause during a background scan still shows the existing "Theme files unavailable" state and blocks file actions
- [ ] #5 Tests cover the stale-result case and the pause case, and the existing picker tests pass without relying on a synchronous scan
<!-- AC:END -->
