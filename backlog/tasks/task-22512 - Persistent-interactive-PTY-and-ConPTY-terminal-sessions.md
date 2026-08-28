---
id: TASK-22512
title: Persistent interactive PTY and ConPTY terminal sessions
status: In Progress
assignee: []
created_date: '2026-08-27 05:01'
updated_date: '2026-08-28 00:00'
labels:
  - console
  - terminal
  - security
  - ux
dependencies:
  - TASK-18926
references:
  - backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
  - backlog/decisions/099-persistent-terminal-session-runtime-boundary.md
  - Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md
  - Docs/superpowers/specs/2026-08-28-persistent-terminal-sessions-design.md
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Finalize and independently review the approved persistent-terminal design and ADR before code work.
2. Qualify pyte and Windows pywinpty against the required shell, terminal-protocol, packaging, and real-platform matrices; stop and revisit ADR-099 if the parser boundary fails materially.
3. Implement the app-global session contracts, authorization states, resource limits, and terminal screen model through focused test-driven slices.
4. Add the admission-gated POSIX PTY and Windows ConPTY backends with one authoritative reaper, bounded control channels, and platform-native cleanup evidence.
5. Add the Console Terminal workspace, session controls, focus routing, danger disclosure, navigation survival, and cleanup receipts without giving widgets process ownership.
6. Complete privacy, context-exclusion, hostile-control, output-flood, memory, shutdown, mounted-TUI, and real POSIX/Windows verification.
7. Update Console, Privacy & Security, Tools, configuration, and platform setup documentation; self-review and record final evidence before closeout.

ADR required: yes.
ADR path: `backlog/decisions/099-persistent-terminal-session-runtime-boundary.md`.
Reason: ADR-099 establishes the long-lived PTY/ConPTY ownership, parser and dependency boundary, process-lifetime state, user-only privacy contract, and cross-platform cleanup semantics introduced by this task.
<!-- SECTION:PLAN:END -->
