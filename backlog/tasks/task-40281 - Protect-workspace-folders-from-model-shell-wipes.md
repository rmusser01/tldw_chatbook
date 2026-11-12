---
id: TASK-40281
title: Protect workspace folders from model shell wipes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - security
  - tools
dependencies: []
references:
  - backlog/tasks/task-25905 - Raw-shell-unbypassable-hardline-command-floor.md
documentation:
  - Docs/superpowers/specs/2026-11-11-workspace-shell-deletion-guard-design.md
  - Docs/superpowers/plans/2026-11-11-workspace-shell-deletion-guard.md
  - backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace chats need to use the raw shell without accidentally deleting or emptying the folders bound to their workspace. Keep ordinary edits and targeted deletions available. The guard covers recognizable local shell commands and applies even when the chat already has a raw shell session grant.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Recognizable Bash, PowerShell and cmd commands that delete a local binding, delete a parent containing it, or wipe the binding are refused before a shell launches.
- [ ] #2 Every local binding of the chat's owning workspace is protected, including read-only and missing folders. A session grant or earlier one-use approval cannot override the guard.
- [ ] #3 Ordinary edits, targeted deletions and user-authored shell commands retain their existing behavior.
- [ ] #4 ADR-094 and the architecture documentation explain what the guard recognizes and where static detection stops.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-094 and test the bounded shell command detector.
2. Enforce the detector before approval and launch using live bindings from the owning workspace.
3. Run the targeted shell suites and Ruff checks, then review the diff.

ADR required: yes
ADR path: backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
Reason: This extends the existing model raw shell refusal policy.
<!-- SECTION:PLAN:END -->
