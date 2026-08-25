---
id: TASK-21140
title: 'Wizard step errors: always-visible surface'
status: To Do
assignee: []
created_date: '2026-08-25 06:14'
labels:
  - ux
  - wizard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings W-1, G-3, plus the hidden show_step_error surface (findings.md sections W/G and the F-1 investigation): step errors render into a .setup-step-error Static at the bottom of an overflowing scroll region, invisible at common terminal sizes; the empty strip paints an error background on Welcome; the Notes step uses the error slot for neutral info (red-on-maroon reassurance); the error copy references a nonexistent Skip-this-step control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A step commit failure is visible without scrolling at 140x40 and 80x24 (error surface pinned near the footer)
- [ ] #2 Empty error surfaces paint no background anywhere in the wizard
- [ ] #3 Notes-step informational text renders in neutral styling and survives a real error being shown
- [ ] #4 No error copy references controls that do not exist
<!-- AC:END -->
