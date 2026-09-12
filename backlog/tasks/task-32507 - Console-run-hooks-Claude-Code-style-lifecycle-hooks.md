---
id: TASK-32507
title: Console run hooks (Claude Code-style lifecycle hooks)
status: Done
assignee: []
created_date: '2026-09-12 15:39'
updated_date: '2026-09-12 16:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-configured external commands at six Console session/run lifecycle events, per spec 2026-09-11-console-run-hooks-design and ADR-148
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Six lifecycle events fire at their production seams, including manual-only prompts, dispatched-only tool results, all approval kinds, queued terminal turns and child settlement.
- [x] #2 Tool guards preserve per-call refusals through reviewer failure and approval exemptions and cannot grant tool permission.
- [x] #3 Hook subprocesses use argv and validated cwd, bounded capture and timeout cleanup; notification admission and allocation are bounded.
- [x] #4 Runtime owns one synchronized engine with live configuration and closes queued and active work at shutdown.
- [x] #5 Prompt refusal preserves retry input and releases preparation; accepted context and Stop ownership survive durable postcommit failure and recovery.
- [x] #6 User guide, spec, ADR-148 and review dispositions match verified behavior; targeted tests and derived-artifact checks pass with explicit baseline limitations.
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
Implemented all six lifecycle hooks with per-call restriction-only guards, bounded capture and notification/approval allocation, synchronized runtime ownership, durable prompt-context recovery and exactly-once terminal/approval coverage. Rebased onto dev eab1188011, preserving release0.2.1 and appearance behavior; resolved colliding documentation task IDs with provenance. ADR required:no new ADR; amended ADR148 and applied existing ADR097 to defer startup imports. Full findings and exact21 baseline exclusions: Docs/superpowers/reviews/2026-09-12-pr-2645-run-hooks.md. Verification:1061 targeted tests passed; final summary fix131 focused tests; startup fix111; latest release rebase127 startup/hook/config/smoke tests. All applicable GitHub checks passed on prior head5151342d69; final rebased-head CI is the merge gate. New module/test Ruff and formatting, changed-production undefined-name checks and whitespace checks passed. All derived artifacts reproduce; task-ID and diagnostic checks reverified after final rebase. Diagnostic review covers28 engine plus6 integration statements; bounded nonblocking output is intentional, raw argv/config/exception content excluded, no new sink. Independent engine/integration reviews found no blockers. Added a lesson proving fail-closed assertions with a successful control on the real async scheduler path.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from TASK-32480 during PR #2645 rebase on 2026-09-12 because the older task on dev retains that ID. New ID checked across remote refs and registered worktrees.
