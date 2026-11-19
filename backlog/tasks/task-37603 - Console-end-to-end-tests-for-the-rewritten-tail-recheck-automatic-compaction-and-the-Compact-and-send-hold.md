---
id: TASK-37603
title: >-
  Console: end-to-end tests for the rewritten-tail recheck, automatic compaction
  and the Compact-and-send hold
status: To Do
assignee: []
created_date: '2026-10-29 09:52'
labels:
  - console
  - trace
  - compaction
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Coverage gaps named by the TASK-33621.29 / TASK-33621.7 review (2026-10-29). (1) The dispatch-binding recheck of a rewritten-tail proof fails capture closed when the memory is reset or a replacement row is deleted or reparented between preparation and dispatch; this is unit-tested only, so what the user then sees is unverified. (2) Automatic compaction and the 'Compact and send' hold produce the same rewritten-tail shape as Compact now, but only Compact now is driven end to end and live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An end-to-end test resets the memory (or deletes a replacement row) between preparation and dispatch and asserts the send fails closed with no provider contact and a visible recovery card
- [ ] #2 An end-to-end test drives automatic compaction followed by a captured send and asserts the send reaches the provider with trace capture intact
- [ ] #3 An end-to-end test drives the 'Compact and send' hold and asserts the same
<!-- AC:END -->
