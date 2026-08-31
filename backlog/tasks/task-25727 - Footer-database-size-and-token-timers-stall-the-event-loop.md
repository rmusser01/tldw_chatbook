---
id: TASK-25727
title: Footer database size and token timers stall the event loop
status: To Do
assignee: []
created_date: '2026-08-31 05:09'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The application log records event loop stalls above one second with the footer database size and token polling timers active. In a keyboard-driven terminal application a stall of that length buffers keystrokes and delivers them late, which is the same input-integrity failure class addressed in earlier Console work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Footer polling work runs off the event loop
- [ ] #2 No footer-attributed stall exceeds the diagnostic threshold under normal use
- [ ] #3 Typing remains responsive while footer counters refresh
<!-- AC:END -->
