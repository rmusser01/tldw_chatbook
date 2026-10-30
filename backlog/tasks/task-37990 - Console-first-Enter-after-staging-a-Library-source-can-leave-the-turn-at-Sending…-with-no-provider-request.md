---
id: TASK-37990
title: >-
  Console: first Enter after staging a Library source can leave the turn at
  'Sending…' with no provider request
status: To Do
assignee: []
created_date: '2026-10-30 04:41'
labels:
  - console
  - staged-context
  - send
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen once, live, while reproducing TASK-33621.8 on origin/dev ed7481a30c (2026-10-29; Anthropic claude-haiku-4-5 through the review proxy, isolated profile, 235x52). Steps: create a Library note, run Search Library so one source is staged, type a question, press Enter. The user row showed 'Sending…' for more than 3 minutes. The proxy saw no provider request, nothing was persisted, and the UI stayed responsive: the composer still held the draft and the '1 staged' badge remained. Clicking Send then worked. The next staged send, on the TASK-33621.8 build, worked with Enter, so this is not reproduced yet.

Why it matters: a send that silently never leaves looks identical to a slow provider, and the only recovery found was guessing that the Send button behaves differently from Enter.

Rule out the driver first: backlog/docs/lessons-live-verification.md records that a tmux driver typing straight after Enter gets sends refused, and that a key burst can lose typed text. Evidence (capture outside the repo): the U06 session evidence folder, shot 01-dev-staged-send-stuck.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A live run of at least five staged-Library sends submitted with Enter, driven per lessons-live-verification (wait for the send acknowledgement, count provider requests at the proxy), either reproduces the stall or rules it out, with the result recorded in this task
- [ ] #2 If reproduced: Enter after staging a Library source either dispatches the request or shows a recoverable state that names what is holding the send; it never sits at 'Sending…' with no provider request
- [ ] #3 If reproduced: a regression test drives the real send path for a staged-Library turn and fails on the pre-fix code for the identified cause
<!-- AC:END -->
