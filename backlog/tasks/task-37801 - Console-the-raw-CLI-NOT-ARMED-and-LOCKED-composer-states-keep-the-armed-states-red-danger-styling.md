---
id: TASK-37801
title: >-
  Console: the raw-CLI NOT ARMED and LOCKED composer states keep the armed
  state's red danger styling
status: To Do
assignee: []
created_date: '2026-10-29 17:13'
labels:
  - console
  - raw-cli
  - ux-review-2026-09-29
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33622.9 made an unarmed or Locked raw-CLI draft read 'RAW CLI · NOT ARMED' / 'Arm…' or 'RAW CLI · LOCKED' / 'Locked' instead of the armed 'RAW CLI · HOST ACCESS' / 'Run'. The text now differs, but the composer and status row keep the same red console-raw-cli-danger styling in all three states, and the original review finding was that the states look identical. A distinct, quieter style for the two states that cannot run a command would let a user tell them from the armed state at a glance, without reading the label.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With raw CLI not armed or Locked, a typed '! ' draft's composer and status row render with a style distinct from the armed state's danger styling, in both light and dark themes
- [ ] #2 The armed state keeps its danger styling
- [ ] #3 The boot CSS budget and the ui-ready module census are not raised
<!-- AC:END -->
