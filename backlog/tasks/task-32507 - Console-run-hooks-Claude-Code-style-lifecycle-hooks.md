---
id: TASK-32507
title: Console run hooks (Claude Code-style lifecycle hooks)
status: In Progress
assignee: []
created_date: '2026-09-12 15:39'
updated_date: '2026-09-12 16:56'
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
Implemented all six run-hook events with restriction-only per-call guards, bounded subprocess capture and notification/approval allocation, synchronized shutdown ownership, durable prompt context/recovery, and exactly-once approval/terminal lifecycle coverage. Rebased onto dev 71313cccd8; retained dev appearance implementation/tests. ADR required: no new ADR; amended backlog/decisions/148-console-run-hooks.md. Full dispositions: Docs/superpowers/reviews/2026-09-12-pr-2645-run-hooks.md. Verification: 1061 targeted tests passed with 21 exact-dev baseline failures explicitly excluded; final summary repair passed 131 focused tests. New modules/tests Ruff and formatter, changed-production undefined-name checks and whitespace checks passed. All derived artifacts reproduce including pinned Mermaid assets. Reviewed 28 engine plus six integration diagnostic statements: bounded nonblocking output is intentional under ADR148; metadata excludes raw argv/config/exception content and no sink topology changed. Renumbered colliding PR task IDs to32506/32507 with provenance. Added the fail-closed scheduler-control testing lesson. Independent engine and integration reviews found no remaining blockers. GitHub checks remain the merge gate.

GitHub Perf Guard found one eager hook module beyond ADR097 startup budget (974/973). Reopen to defer hook helpers until event use, verify real UI-ready census and hook regressions, then publish and wait for all CI checks.

CI startup regression fixed under existing ADR097 by deferring controller hook imports to lifecycle use; added run_hooks to the real UI-ready absence assertion without changing the budget. Real startup census plus engine/controller/config regressions:111 passed. Independent diff review found no behavior change. Latest dev ea406f4d41 adds only branch-protection documentation.

Release PR2588 landed during CI with a conflicting documentation task ID. Preserve its existing ID, renumber the unrelated fallback-model design task with fresh remote/worktree provenance, and rebase onto release dev eab1188011 before the final merge gate.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from TASK-32480 during PR #2645 rebase on 2026-09-12 because the older task on dev retains that ID. New ID checked across remote refs and registered worktrees.
