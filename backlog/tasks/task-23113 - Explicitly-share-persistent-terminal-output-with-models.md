---
id: TASK-23113
title: Explicitly share persistent terminal output with models
status: To Do
assignee: []
created_date: '2026-08-28 00:00'
updated_date: '2026-08-28 00:00'
labels:
  - console
  - terminal
  - tools
  - privacy
  - security
dependencies:
  - TASK-22512
references:
  - backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
  - backlog/decisions/099-persistent-terminal-session-runtime-boundary.md
  - Docs/superpowers/specs/2026-08-28-persistent-terminal-sessions-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user deliberately share bounded persistent-terminal output with a model, either by attaching a selected excerpt to the next message or by approving a model request to read a session snapshot, while terminal contents remain private and model-inaccessible by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Persistent-terminal input, output, and metadata remain excluded from model context and model tools unless the user performs an explicit sharing action.
- [ ] #2 A user can select a bounded terminal excerpt, preview exactly what will be shared, and attach it to Next Send without granting continuing terminal access.
- [ ] #3 A model-facing read capability can request only a bounded snapshot from an identified live terminal session and cannot write input, create sessions, close sessions, or invoke a shell command.
- [ ] #4 Every model snapshot request requires a request-visible user decision that identifies the session and requested range; no persistent or session-wide silent Allow state is accepted.
- [ ] #5 Shared terminal content is decoded, control-sequence-safe, size-bounded, and represented without allowing terminal escape sequences or markup to affect unrelated UI.
- [ ] #6 Closing a session, disarming Terminal, or exiting Chatbook revokes pending and future snapshot access without changing the one-shot raw CLI, shell_exec, or virtual_cli permission contracts.
- [ ] #7 Transcript, persistence, export, and diagnostics behavior clearly distinguishes user-attached excerpts from approval-gated model snapshots and does not duplicate unshared terminal content.
- [ ] #8 Focused permission, privacy, race, truncation, Unicode, hostile-control-sequence, session-close, and restart tests cover both sharing paths.
- [ ] #9 A design specification and ADR decision are approved before implementation because this introduces a new model-data and terminal-privacy boundary.
- [ ] #10 User documentation explains that terminal sharing is opt-in, bounded, potentially sensitive, and never grants the model interactive terminal control.
<!-- AC:END -->
