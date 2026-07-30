---
id: TASK-1479
title: >-
  Evals export dialog: filename input unusable, Enter navigates instead of confirming
status: To Do
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by live UAT (2026-07-30) at 235x52. Export (`e`) opens `Third_Party.textual_fspicker.FileSave`, where three failures compounded: the filename input rendered ~5 cells wide and empty (the pre-filled default `<bench>.json` was invisible), Enter with list focus navigated directories (activated `..`) instead of confirming, and text typed after clicking the input never appeared. A keyboard-driven export could not be completed at all — despite `e` being a keyboard affordance advertised in the footer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The filename input is wide enough to show a typical default filename, and the pre-filled default is visible on open
- [ ] For a save dialog, initial focus lands in the filename input and Enter there confirms the save
- [ ] The keyboard-only path `e` then Enter exports with the default name into the dialog's directory
- [ ] Directory navigation in the file list still works (Enter on a directory still navigates)
- [ ] A test or scripted verification covers the keyboard-only export path
<!-- AC:END -->
