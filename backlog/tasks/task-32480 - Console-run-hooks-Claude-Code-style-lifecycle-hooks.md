---
id: TASK-32480
title: Console run hooks (Claude Code-style lifecycle hooks)
status: Done
assignee: []
created_date: '2026-09-12 15:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-configured external commands at six Console session/run lifecycle events, per spec 2026-09-11-console-run-hooks-design and ADR-148
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Six events fire from their seams (UserPromptSubmit wake-exempt, PreToolUse deny-only fail-closed, PostToolUse dispatched-only with run_id, ApprovalRequested exactly-once per round incl. detached, Stop terminal exactly-once incl. queue-chain twin, SubagentStop wrapping settle),Engine: argv-only subprocesses, JSON stdin envelope, deny-only verdicts, per-purpose fail direction, process-group timeout kill, 4000-char budget, never raises,ConsoleRuntime-owned engine singleton, headless-reachable, live config per fire,User-scope [hooks] config with fail-loud validation; user guide + AGENTS.md docs; ADR-148 Accepted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on feat/console-run-hooks (187a1892d0..af084528a7) via SDD: 9 tasks, per-task reviews + final whole-branch review, 32 coordinator rulings recorded in the plan workspace ledger. 272+ tests green; 5 named pre-existing base failures unrelated (verified at base in a throwaway worktree). Follow-ups deferred to the settings sub-screen PR: session-root cwd threading (engine cwd= is the forward contract), bounded notify queue, queue-chain Stop twin test.
<!-- SECTION:NOTES:END -->
