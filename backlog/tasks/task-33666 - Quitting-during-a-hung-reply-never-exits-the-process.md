---
id: TASK-33666
title: Quitting during a hung reply never exits the process
status: To Do
assignee: []
created_date: '2026-10-02 19:30'
labels:
  - console
  - lifecycle
  - bug
dependencies: []
references:
  - qa/task-33662-recovery-relaunch/README.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: found during TASK-33662's live check on 2026-10-02; the bug predates that change.

With a reply hung, Ctrl+Q → Quit settles the reply as `stopped`, but the process stays alive after the app unmounts:
- on the TASK-33662 build, for more than 2 minutes, even after the stub provider was stopped;
- on the merge-base build, still alive after 40 s.

Users must kill the terminal to get out. Capture q1 in qa/task-33662-recovery-relaunch/ shows the quit dialog during a live run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a reply hung (a provider that accepts the request and never streams), Ctrl+Q → Quit exits the process within a bounded time (state the bound) and the reply is settled as stopped.
- [ ] #2 The cause, meaning which thread or task keeps the process alive, is named in the notes with evidence.
- [ ] #3 A test reproduces the hang and passes after the fix.
<!-- AC:END -->
