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
  - Docs/superpowers/plans/2026-08-28-persistent-terminal-sessions-implementation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a deliberately separate, user-controlled interactive terminal experience for workflows that cannot fit the one-shot raw CLI contract, without weakening or silently extending the raw CLI or model shell execution boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Interactive terminal work remains a separate feature and does not change the one-shot raw CLI or model shell_exec contracts.
- [x] #2 A user can create, name, list, focus, and close persistent terminal sessions from a dedicated Console surface.
- [x] #3 POSIX sessions use an admitted controlling PTY; Windows launches fail closed without pywinpty, legacy winpty, or ordinary-pipe fallback until a new or superseding ADR qualifies a replacement boundary.
- [x] #4 Each running session retains its own shell process, current directory, environment, and terminal state until the shell exits, the user closes it, or Chatbook exits; an ordinary shell exit retains final terminal state until the user closes the record.
- [x] #5 Terminal input and output support interactive programs, resize events, Unicode, and bounded scrollback without leaking terminal controls into unrelated UI.
- [x] #6 The UI makes full OS-user authority and the absence of workspace confinement unmistakable before a session starts.
- [x] #7 Session concurrency and resource limits are explicit and produce predictable refusal behavior.
- [x] #8 Stop, shutdown, app-process-failure cleanup, descendant cleanup, and honest cleanup-unproven behavior have focused POSIX and Windows verification.
- [x] #9 The implementation task records an ADR decision before code work because this introduces a long-lived runtime and UX boundary.
- [x] #10 User documentation distinguishes persistent terminals from raw ! commands, model shell_exec, and virtual_cli.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Execute the owner-approved detailed implementation plan against the independently reviewed persistent-terminal design and accepted ADR-099 boundary.
2. Qualify pyte, the terminal-specific scrubbed environments, and Windows pywinpty against the required shell, terminal-protocol, packaging, and real-platform matrices; record results in `Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md`, fail the affected backend if a mandatory row fails, and require a new or superseding ADR decision before changing the pinned dependency/API boundary.
3. Implement the app-global session contracts, authorization states, resource limits, and terminal screen model through focused test-driven slices.
4. Add the admission-gated POSIX PTY backend with one authoritative reaper and bounded control channels; keep Windows unavailable and fail closed until a replacement ConPTY boundary passes native qualification under a new or superseding ADR.
5. Add the Console Terminal workspace, session controls, focus routing, danger disclosure, navigation survival, and cleanup receipts without giving widgets process ownership.
6. Complete privacy, context-exclusion, hostile-control, output-flood, memory, shutdown, mounted-TUI, real POSIX verification, and native Windows fail-closed verification.
7. Update Console, Privacy & Security, Tools, configuration, and platform setup documentation; self-review and record final evidence before closeout.

ADR required: yes.
ADR path: `backlog/decisions/099-persistent-terminal-session-runtime-boundary.md`.
Reason: ADR-099 establishes the long-lived PTY/ConPTY ownership, parser and dependency boundary, process-lifetime state, user-only privacy contract, and cross-platform cleanup semantics introduced by this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented the ADR-099 persistent Terminal as an app-global, user-only runtime
that remains separate from one-shot raw CLI and model `shell_exec`. The manager
owns admitted POSIX PTYs, one runtime bridge and authoritative reaper per
session, bounded input/output actors, compact immutable screen snapshots,
retained exited state, and identity-checked cleanup. The Console adds dedicated
session controls, explicit full-host danger disclosure, independent per-launch
arming under the shared saved unlock, and navigation/recompose survival without
widget process ownership.

macOS ARM64 and Ubuntu 24.04 ARM64 passed mounted real-shell, cleanup/crash,
packaging, four-session memory, and ANSI-latency qualification. Native Windows
qualification rejected the pinned pywinpty boundary, so Windows remains
content-free fail closed and the distribution ships no Windows fallback. Final
review also hardened transient PTY `EIO` handling and Textual recompose gaps.
User documentation now covers authority, profiles and side effects, limits,
keys, cleanup, privacy, persistence, and platform support.

Verification used focused reachable suites plus native macOS ARM64 and Ubuntu
24.04 ARM64 mounted, cleanup, packaging, resource, and latency matrices. The
user explicitly declined the optional full repository suite on 2026-09-01;
the evidence artifact records that choice and does not claim a full-suite run.

ADR required: yes.
ADR: `backlog/decisions/099-persistent-terminal-session-runtime-boundary.md`.
Evidence:
`Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md`.
