---
id: TASK-37463
title: 'Console: a staged attachment chip hides the composer draft at 80x24'
status: To Do
assignee: []
created_date: '2026-10-29 07:49'
labels:
  - console
  - p2
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while live-verifying TASK-33620.2 (2026-10-29). At 80x24, once an image is staged the composer's action row (Composer, Menu, the attachment chip, Send, Dictate, clear) takes the whole width and the draft text is not painted at all: typed text goes into the draft (it is sent) but the user cannot see it. Seen on a dev build captured 2026-10-27 (evidence tools/evidence/bd5/U14/03-dev-80x24-image-staged-no-reason) and on the TASK-33620.2 branch (30, 55); the branch does not touch the composer row's width budget. TASK-24415 and TASK-24620 fixed the same starvation for the send-disabled reason and the voice chip; the attachment chip is the remaining case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With an image staged at 80x24, the draft text is visible on the rendered screen while the chip stays readable
- [ ] #2 A test paints the composer at 80x24 with a staged image and a draft and asserts both are visible, failing on the pre-fix code
<!-- AC:END -->
