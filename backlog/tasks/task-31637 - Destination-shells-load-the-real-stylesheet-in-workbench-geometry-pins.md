---
id: TASK-31637
title: 'Destination shells: load the real stylesheet in workbench geometry pins'
status: To Do
assignee: []
created_date: '2026-09-05 08:41'
labels:
  - testing
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The DestinationHarness used by the destination workbench geometry tests mounts screens without the app stylesheet, so every width/height/visibility assertion it makes is measured against Textual defaults rather than the CSS that actually ships. The pins therefore pass whether or not the layout is correct, and cannot catch the regressions they were written for. Pre-existing across the sibling destinations, surfaced by the meeting-transcription whole-branch review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The harness renders destination screens with the same stylesheet the running app applies
- [ ] #2 At least one existing geometry pin is shown to fail when its screen's CSS rule is removed
<!-- AC:END -->
