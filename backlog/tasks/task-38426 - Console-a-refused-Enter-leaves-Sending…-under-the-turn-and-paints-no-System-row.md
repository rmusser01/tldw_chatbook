---
id: TASK-38426
title: >-
  Console: a refused Enter leaves 'Sending…' under the turn and paints no System
  row
status: To Do
assignee: []
created_date: '2026-10-31 03:50'
labels:
  - console
  - errors-recovery
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found live while verifying TASK-33621.11's review fixes, and identical on dev 7536ac3b87 (evidence U09/30). With a PNG staged on claude-haiku-4-5 (a model the capability table says cannot take images), typing a message and pressing Enter shows the user row with 'Sending…' under it. The send gate refused the turn in 85 ms (console_send_stage phase=ui_dispatch status=refused), but the 'Sending…' acknowledgement stays and the refusal's System row is not painted. It appears only after a later sync, and then above the stale 'Sending…' turn. The user sees a send that looks in flight and never learns why it stopped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A refused Enter removes or resolves its 'Sending…' acknowledgement at once
- [ ] #2 The refusal's System row is painted when the refusal happens, below the refused turn
- [ ] #3 A mounted test drives a refused Enter through the real keyboard path and asserts both
<!-- AC:END -->
