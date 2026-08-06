---
id: TASK-2710
title: Audit remaining super().on_mount() calls over BaseAppScreen
status: To Do
assignee: []
created_date: '2026-08-06 20:35'
labels:
  - ui
  - tech-debt
dependencies:
  - task-2610
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2610's root cause: Textual dispatches EVERY `on_mount` along the MRO for one Mount
event, so a subclass handler calling `super().on_mount()` runs the parent handler twice.
That crashed Lab ▸ Speech because `LabFrameScreen.on_mount` mounts widgets. Roughly twenty
other screens/widgets still call `super().on_mount()` over `BaseAppScreen` (grep
`super().on_mount()` under `tldw_chatbook/`); today that only duplicates the base's log
line, so they are harmless-by-luck — but the moment anyone adds real work to
`BaseAppScreen.on_mount` (or to any intermediate class), every remaining call site
detonates the same way. `BaseAppScreen.on_mount`'s docstring now states the contract;
this task removes the latent calls so the contract is enforced by absence, not by
discipline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No `super().on_mount()` call remains in tldw_chatbook code whose parent chain includes a class that defines `on_mount` (Third_Party/ excluded)
- [ ] #2 Each removal is verified not to change behavior (the parent handler still runs exactly once via the dispatcher)
- [ ] #3 A guard (test or lint rule) prevents new `super().on_mount()` calls from being introduced over BaseAppScreen
<!-- AC:END -->
