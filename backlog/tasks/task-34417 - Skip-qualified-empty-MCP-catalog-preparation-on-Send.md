---
id: TASK-34417
title: Skip qualified empty MCP catalog preparation on Send
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 07:22'
updated_date: '2026-10-06 15:50'
labels:
  - console
  - performance
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce avoidable Send preparation when the captured maximum permits no MCP tools, while retaining enabled capabilities and current ownership and permission behavior. Immediate terminal feedback and sub-second application dispatch remain the wider performance objective.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A qualified stock Send with an exact empty frozen MCP maximum avoids provider construction and catalog source reads.
- [x] #2 Empty live composition clears earlier MCP inspector counts while disposable preview leaves them unchanged.
- [ ] #3 Unset, nonempty, custom and plugin paths preserve existing validation, permission, ownership, disconnect and kill-switch behavior.
- [ ] #4 Affected source-change, ownership, cancellation and close regressions pass without weakening original guards.
- [ ] #5 Original operation counts and targeted verification are recorded; the remaining feedback and dispatch gap remains explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
Reason: a stock early return preserves existing interfaces and finite custody; hook reconciliation remains unchanged under ADR-197.
1. Prepare tests on the committed stock MCP foundation; repeat on integrated current main fixes before production code.
2. Add and run original-code count, live/preview transition, unset/nonempty, custom, plugin and source/owner controls.
3. Add the single qualified empty-maximum early return after current-source checks.
4. Run the directly affected regression selection and scoped lint/format/diff checks.
5. Record original count changes and remaining timing gaps; preserve all current guards and native retirement checks.
Approved spec: Docs/superpowers/specs/2026-10-05-incremental-send-speed-design.md
Approved detailed plan: Docs/superpowers/plans/2026-10-06-incremental-send-speed-first-slice.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the qualified stock empty-MCP early return after original source/current-owner checks. Exact empty frozen maxima skip provider construction and catalog preparation; plugin maxima and custom/unset/nonempty routes keep ordinary behavior. No hook, worker, history, schema, dependency or permission interface changed.
Native Windows original counts: dispatch and preview each 1 provider construction / 1 catalog preparation / 1 native catalog read -> 0/0/0. Focused tests: 11 pass after verified RED on eeaee. Full seven-file candidate: 321 pass, 3 failures. Matching unchanged collection: 318 pass, 6 failures (the same 3 baseline failures plus 3 expected missing-optimization failures). The writable-selection case passes alone on both versions; two cancellation failures reproduce in both isolated versions. No new regression shown.
New-test Ruff and diff checks pass; controller has identical 55 preexisting Ruff findings, no new finding. New-test-range formatting preserved AST. Self-review confirms source fences precede the guard and plugin/custom behavior remains covered.
Base: eeaee0328d (includes ca31b1c9f7 dev integration). Existing ADR-126/134/197 govern unchanged lifetimes; no new ADR required.
Modified: Chat/console_chat_controller.py, Tests/MCP/test_console_snapshot_source_contracts.py; verification record Docs/Development/2026-10-06-incremental-send-speed-first-slice.md.
Task remains In Progress while supported-host plugin evidence and actual Send timing/feedback acceptance are gathered. The overall 100 ms acknowledgment and one-second dispatch targets are not yet established.
<!-- SECTION:NOTES:END -->
