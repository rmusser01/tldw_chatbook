---
id: TASK-36729
title: Plan shared served Console ownership and mobile supervision implementation
status: Done
created_date: 2026-10-25 17:28
dependencies:
- TASK-36728
assignee:
- '@codex'
labels:
- console
- web-server
- mobile
- planning
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
updated_date: 2026-10-25 18:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the accepted shared Console/mobile supervision design into a source-anchored implementation sequence that can be executed and reviewed without reopening the rejected mobile storage and runtime split.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A saved implementation plan maps every approved specification requirement to named implementation tasks, exact source/test paths, public interfaces, concrete code/test steps and targeted verification commands.
- [x] #2 Atomic outcome-based Backlog tasks are created in dependency order, reference only existing earlier task IDs and link ADR-244 and the implementation plan.
- [x] #3 The plan preserves all seven design-review corrections, migration and authentication boundaries, and distinguishes automated, served-browser and actual phone/VPN evidence.
- [x] #4 The plan passes self-review for coverage, interface consistency, placeholders, links and task metadata; no product code or real user data changes during planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation planning under the existing accepted ADR; no new ownership or security contract is introduced.

1. Map the approved spec, governing ADRs, source seams, connected callers and existing test fixtures.
2. Decompose implementation into independently testable outcomes and create Backlog tasks in dependency order without forward task references.
3. Write the implementation plan with exact files, cross-task interfaces, concrete test/code steps, targeted commands, delivery gates and a spec-coverage map.
4. Self-review requirement coverage, interface names, placeholders, local links and task structure; preserve unrelated work and commit only planning artifacts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the source-anchored implementation plan for the accepted design under ADR-244 and 11 atomic To Do implementation tasks in dependency order. The plan covers physical execution fencing, independent control/event draining, retained hosts and terminal modes, command-owned input and exact controls, runtime decisions/projections, durable ticket/outcome reconciliation, coherent legacy preview/import/quarantine, canonical API cutover, phone UI and real served/browser qualification.

Mapped every spec section and all seven review corrections to tasks and evidence gates. Defined public interfaces, source/test files, concrete code/test steps, targeted commands and caller/fixture updates, including old Canvas control-loss assumptions. Self-review corrected ticket issuance/query identity, per-child identity on replacement, explicit native input revisions and recovery-state mapping. Existing ADR-244 applies; no new ADR is required and no product behavior is claimed implemented.

Validated all 12 new task records and their existing earlier dependencies/documentation, 37 local links, 25 Python examples with AST parsing, one JavaScript example with Node syntax checking and two SQL examples in isolated in-memory SQLite. Task IDs were checked against locally available reachable refs and 36 worktrees before allocation. Placeholder, scope and interface review passed. Product runtime tests, external providers, real user databases/migration, physical phone/VPN, push and merge were outside this planning task.
Repository-wide ID validation reports existing duplicate IDs 34991 and 35028, each involving unrelated tracked/untracked task files already present in the pre-planning status. None of this plan's 12 new task IDs collides, and scoped task validation passed. Preserved those files rather than renumbering another workstream; the whole-checkout guard is not reported green.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
