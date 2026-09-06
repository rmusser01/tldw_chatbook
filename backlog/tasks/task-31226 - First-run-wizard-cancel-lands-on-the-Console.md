---
id: TASK-31226
title: First-run wizard cancel lands on the Console
status: Done
assignee: []
created_date: '2026-09-04 01:00'
updated_date: '2026-09-06 00:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Esc-exiting the first-run setup wizard leaves the user on Home (the screen it was pushed over). Cancel should drop the user onto the Console workbench, while cancelling a Settings/palette re-run must stay put. Also pins that the workspace tree keeps empty registered workspaces (UAT artifact triage).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cancelling the boot-offered first-run wizard (Esc exit / finish-later) switches to the Console tab,Cancelling a Settings/command-palette wizard re-run leaves the current screen unchanged,Completed wizard results keep routing per their exit_route (regression pin),A registered workspace with zero conversations keeps its tree node (mounted pin)
<!-- AC:END -->
