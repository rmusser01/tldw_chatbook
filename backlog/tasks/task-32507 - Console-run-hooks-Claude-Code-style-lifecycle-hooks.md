---
id: TASK-32507
title: Console run hooks (Claude Code-style lifecycle hooks)
status: In Progress
assignee: []
created_date: '2026-09-12 15:39'
updated_date: '2026-09-12 16:28'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/148-console-run-hooks.md
Reason: repair the existing hook contract on current dev; amend ADR-148 for the restriction-only guard seam and documented bounds.
1. Rebase preserving dev runtime, approval and appearance behavior.
2. Independently review all PR changes and 16 posted findings; record dispositions.
3. Reproduce confirmed defects, then repair deny enforcement, subprocess bounds/cleanup, lifecycle and config ownership.
4. Run targeted integration/architecture gates and independent final review.
5. Push with an explicit lease, check final GitHub status/comments, merge when green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Full PR review repaired permission bypasses, bounded subprocess and notification resources, synchronized shutdown ownership, durable prompt-context recovery and lifecycle/approval coverage. ADR-148 amended without a new architecture decision. Posted findings and independent defects are recorded in Docs/superpowers/reviews/2026-09-12-pr-2645-run-hooks.md. Targeted engine (77) and lifecycle (23) tests pass; final rebase, integration and derived-artifact verification remain in progress. Removed obsolete prior completion claims and renumbered colliding task IDs with provenance.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from TASK-32480 during PR #2645 rebase on 2026-09-12 because the older task on dev retains that ID. New ID checked across remote refs and registered worktrees.
