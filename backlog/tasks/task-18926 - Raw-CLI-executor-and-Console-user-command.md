---
id: TASK-18926
title: Raw CLI executor and Console user command
status: In Progress
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-27 05:16'
labels:
  - console
  - tools
  - security
  - ux
dependencies: []
references:
  - backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md
  - Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give advanced users an unmistakably dangerous direct one-shot host-shell escape hatch from Console without an LLM call. Raw CLI is disabled by default, requires a persisted unlock plus per-launch arming, streams bounded local output, and must state plainly that commands retain the full authority of the OS user running Chatbook.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Persistent raw CLI unlock defaults Off, saved unlock and process-memory-only Arm are separate, and every Chatbook launch starts unarmed
- [ ] #2 The danger UI states full OS-user filesystem, process, and network authority; scrubbed-environment limitations; local log persistence; and best-effort descendant cleanup without calling raw CLI sandboxed or confined
- [ ] #3 Only a physically typed exact exclamation-space prefix selects raw mode; a pasted prefix cannot select it, a typed prefix may be followed by pasted command text, and escaped exclamation-space sends ordinary chat text
- [ ] #4 An armed user command bypasses slash parsing, provider calls, token usage, and the prompt queue, including while a model run is active; locked or unarmed submits fail inline without changing settings
- [ ] #5 One shared one-shot executor supports auto, Bash, PowerShell, and CMD selectors with fixed profile-disabled launch arguments, DEVNULL stdin, a 16 KiB command limit, and a 300 second ceiling
- [ ] #6 The executor starts from an empty environment and copies only the approved shell-essential allowlist; callers cannot inject environment overrides
- [ ] #7 The shell cannot start until the worker is admitted to its owned POSIX process group or Windows Job Object; timeout, Stop, disarm, and shutdown perform bounded cleanup and report whether cleanup was proven
- [ ] #8 Stdout and stderr stream separately into a live Tool-style row with elapsed time and Stop; ANSI, OSC, unsafe controls, memory, transcript preview, disk spool, and update frequency are bounded
- [ ] #9 User commands create durable AgentRunsDB rows with agent_kind local_command and resumable markers while remaining absent from provider history, agent counts, rails, fleet state, and costs
- [ ] #10 Focused tests include real POSIX process-tree evidence, native Windows Job Object evidence, admission-race, output-flood, cancellation, parsing, zero-provider, persistence, and mounted Settings and Console behavior
- [ ] #11 Console, Privacy and Security, configuration, and authority documentation describe syntax, limits, re-arming, local persistence, and the non-sandboxed trust boundary
- [ ] #12 User-command run logs and trajectory data remain unavailable to provider history and model-facing run-log search, slice, and statistics tools.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md
Reason: ADR-093 authorizes the host-authority shell boundary and fixes the false-by-default unlock, per-launch arm, process-ownership, persistence, and no-sandbox contracts.

Detailed plan: Docs/superpowers/plans/2026-08-26-raw-cli-user-command.md

1. Pin request, result, shell argv, scrubbed-environment, and config contracts with focused tests.
2. Build and verify the admitted one-shot executor over the shared process-tree owner.
3. Add app-owned arming, cancellation, and shutdown lifecycle.
4. Add the canonical Privacy & Security danger gate.
5. Track physically typed ! prefix provenance and intercept direct sends before slash/provider/queue seams.
6. Stream bounded output into a stoppable display-only TOOL marker.
7. Persist and restore local_command records while proving provider, agent, fleet, rail, and cost exclusions.
8. Update authority documentation and complete focused POSIX, native Windows, mounted UI, static, and live verification.
<!-- SECTION:PLAN:END -->
