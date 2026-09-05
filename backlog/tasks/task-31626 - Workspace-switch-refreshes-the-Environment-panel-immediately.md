---
id: TASK-31626
title: Workspace switch refreshes the Environment panel immediately
status: To Do
assignee: []
created_date: '2026-09-04 23:10'
labels:
  - console
  - inspector
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Switching workspace (or conversation) while the Console Inspect rail is open
leaves the Environment panel showing the OLD workspace's branch, change
counts, PR and task rows for up to a full 10-second poll interval. The
controller (`UI/Console_Modules/environment.py`) only notices a scope change
on its next `poll_tick`; the scope change itself dispatches nothing, and the
stale-scope guard's only job is to DROP landings that arrive for the previous
root — it never asks for new ones.

The result is a panel that confidently describes the wrong repository. The
spec (TASK-31450) lists "workspace/conversation scope change" as a dispatch
trigger; today it is only a discard trigger.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Changing the active workspace root with the Inspect rail open dispatches a refresh immediately rather than waiting for the next poll tick
- [ ] #2 The panel never shows the previous workspace's data alongside the new one — rows for a superseded root are cleared or replaced, not left standing
- [ ] #3 A scope change while the rail is CLOSED still dispatches nothing (the no-work-while-collapsed guarantee is unchanged)
- [ ] #4 Rapid successive workspace switches settle on the final root's data with no landing from an intermediate root painted
<!-- AC:END -->
