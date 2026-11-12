---
id: TASK-37651
title: Hands-free flashes Starting and Cancel on every loop turn
status: To Do
assignee: []
created_date: '2026-10-29 12:47'
labels:
  - console
  - voice
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-33626.1. The header Hands-free control reads 'Starting…' whenever the loop's capture is in the starting state, and the composer mic reads 'Cancel' for every one-shot dictation start. Each loop turn that reopens the capture, and each ordinary dictation press, therefore flashes 'Starting…' / 'Cancel' for a few hundred milliseconds, and the header reflows by about seven cells each time because the state word is layout-sized. The labels exist to explain a long first-run speech-model load and to offer a way out of it; a sub-second reopen is not that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the speech model is already loaded, reopening the capture between Hands-free turns does not change the header word from 'On' or the mic label from its listening label
- [ ] #2 A first-run or slow model load still shows 'Hands-free: Starting…' and a visible 'Cancel' mic label, as TASK-33626.1 requires
- [ ] #3 The header row does not shift neighbouring controls during a turn-to-turn reopen
- [ ] #4 Tests cover both the fast reopen and the slow first load
<!-- AC:END -->
