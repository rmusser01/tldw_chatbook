---
id: TASK-22512
title: Persistent interactive PTY and ConPTY terminal sessions
status: To Do
assignee: []
created_date: '2026-08-27 05:01'
updated_date: '2026-08-27 05:01'
labels:
  - console
  - terminal
  - security
  - ux
dependencies:
  - TASK-18926
references:
  - backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md
  - Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a deliberately separate, user-controlled interactive terminal experience for workflows that cannot fit the one-shot raw CLI contract, without weakening or silently extending the raw CLI or model shell execution boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Interactive terminal work remains a separate feature and does not change the one-shot raw CLI or model shell_exec contracts.
- [ ] #2 A user can create, name, list, focus, and close persistent terminal sessions from a dedicated Console surface.
- [ ] #3 POSIX sessions use a PTY; Windows sessions use ConPTY or a documented platform-supported equivalent.
- [ ] #4 Each session retains its own shell process, current directory, environment, and terminal state until explicitly closed or Chatbook exits.
- [ ] #5 Terminal input and output support interactive programs, resize events, Unicode, and bounded scrollback without leaking terminal controls into unrelated UI.
- [ ] #6 The UI makes full OS-user authority and the absence of workspace confinement unmistakable before a session starts.
- [ ] #7 Session concurrency and resource limits are explicit and produce predictable refusal behavior.
- [ ] #8 Stop, shutdown, crash recovery, and orphan cleanup have focused POSIX and Windows verification.
- [ ] #9 The implementation task records an ADR decision before code work because this introduces a long-lived runtime and UX boundary.
- [ ] #10 User documentation distinguishes persistent terminals from raw ! commands, model shell_exec, and virtual_cli.
<!-- AC:END -->
