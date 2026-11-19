---
id: TASK-38400
title: >-
  Console: pin the Temporary segment's capture state on a recreated Console's
  first frame
status: To Do
assignee: []
created_date: '2026-10-31 01:38'
labels:
  - console
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-37840 passes a Temporary chat's resolved capture into ConsoleStatusChips at compose, so a recreated Console (restore_state, screen recreation) paints 'Temporary · not saved · capture on' on its first frame. No test covers that first frame: with the compose argument mutated to False the suite stays green, because the next full sync pass repaints the segment. The activation-path tests in test_console_composer_menu have the same blind spot for their direct pushes now that every full sync pass re-pushes the segment. Raised by the pre-merge review of TASK-37840.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test recreates the Console while a Temporary chat has Capture On chosen and reads the segment before any full sync pass runs, and it fails with the compose-time capture state dropped
- [ ] #2 The test passes under the plain pytest command with the bootstrap_profile marker
<!-- AC:END -->
