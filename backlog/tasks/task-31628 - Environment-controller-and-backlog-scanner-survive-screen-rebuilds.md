---
id: TASK-31628
title: Environment controller and backlog scanner survive Console screen rebuilds
status: To Do
assignee: []
created_date: '2026-09-04 23:10'
labels:
  - console
  - inspector
  - performance
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConsoleEnvironmentController` — and the `BacklogTaskScanner` mtime cache it
owns — are constructed in `ChatScreen.__init__`, so their lifetime is one
screen instance. A new `ChatScreen` is built on every visit to the Console
tab (the carried ledger item from the 2026-08-29 holistic performance
review), which means every visit throws away a warm snapshot and a warm
backlog cache and pays the cold scan again. On this repository that is a full
frontmatter parse of thousands of task files, repeated per navigation, for
data that did not change.

The same lifetime problem also means a landing in flight when the screen is
rebuilt has nowhere to land, and the panel starts from "No git workspace"
each time rather than repainting what it already knew.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Navigating away from the Console tab and back does not re-run a cold backlog scan — the mtime cache and the last snapshot survive the screen rebuild
- [ ] #2 Returning to the Console with the Inspect rail open repaints the last known Environment data immediately, rather than showing the empty state until the next poll
- [ ] #3 A landing that arrives across a screen rebuild is either applied to the live screen or dropped cleanly — never applied to a dead one
- [ ] #4 The no-work-while-collapsed and no-work-while-off-tab guarantees still hold: a longer-lived controller must not poll for a screen nobody is looking at
<!-- AC:END -->
