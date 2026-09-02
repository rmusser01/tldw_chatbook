---
id: TASK-26034
title: Wire the Terminal package to a Console pane
status: To Do
assignee: []
created_date: '2026-08-31 15:47'
labels:
  - console
  - ux
  - tools
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A fully built terminal implementation ships and is unreachable. Verified on origin/dev: the Terminal/ package exists with a session manager and screen model, but a named grep for TerminalBackend, TerminalSessionManager and tldw_chatbook.Terminal across UI/ and Widgets/ returns only one unrelated comment about Terminal.app key handling - no UI mounts it. Users shell out to another window instead. Hermes offers an inline shell in the composer that costs no tokens. This is wiring existing code to a pane, not building a terminal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A terminal pane is reachable from the Console and runs an interactive shell session
- [ ] #2 The pane is backed by the existing Terminal package rather than a new implementation
- [ ] #3 Terminal sessions are user-driven only: the agent cannot dispatch into this pane, which stays behind the raw-shell tool and its approval gate
- [ ] #4 Session lifecycle is bounded - closing the pane terminates the process group, and no session survives application exit
- [ ] #5 The pane honors the Console keyboard model and does not capture keys the shell needs
- [ ] #6 If the Terminal package cannot start a session on the platform, the pane reports why rather than appearing broken
<!-- AC:END -->
