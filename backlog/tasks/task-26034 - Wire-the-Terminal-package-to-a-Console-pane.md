---
id: TASK-26034
title: Wire the Terminal package to a Console pane
status: Done
assignee: []
created_date: '2026-08-31 15:47'
updated_date: '2026-09-02 06:17'
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
- [x] #1 A terminal pane is reachable from the Console and runs an interactive shell session
- [x] #2 The pane is backed by the existing Terminal package rather than a new implementation
- [x] #3 Terminal sessions are user-driven only: the agent cannot dispatch into this pane, which stays behind the raw-shell tool and its approval gate
- [x] #4 Session lifecycle is bounded - closing the pane terminates the process group, and no session survives application exit
- [x] #5 The pane honors the Console keyboard model and does not capture keys the shell needs
- [x] #6 If the Terminal package cannot start a session on the platform, the pane reports why rather than appearing broken
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Superseded by dev's TASK-22512 (persistent terminal sessions, PRs #2299 +
#2270, merged 2026-09-01/02) before this task started. Verified against the
merged surface, AC by AC: pane reachable from the Console
(UI/Console_Modules/terminal.py ConsoleTerminalController + the
console_terminal_workspace/session_modal widgets) — AC#1; backed by the
existing Terminal package (session_manager/launch/screen_model/protocol_gate)
— AC#2; explicitly separated from the one-shot raw CLI and model shell_exec
contracts (22512 AC#1), so the agent cannot dispatch into it — AC#3; bounded
lifecycle with app-shutdown wiring and cleanup verification (22512 AC#4/#8) —
AC#4; interactive input/resize/Unicode/scrollback without leaking controls
(22512 AC#5) — AC#5; POSIX-only with fail-closed Windows behavior and an
honest 'persistent Terminal backend unavailable' error (22512 AC#3 +
app.py:_build_terminal_backend) — AC#6. Nothing left to build; #2270
additionally deferred the terminal UI off the boot path (ADR-097).
<!-- SECTION:NOTES:END -->
