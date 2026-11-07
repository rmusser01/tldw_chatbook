---
id: TASK-38523
title: Plan native shared-worktree workflow implementation
status: Done
created_date: 2026-11-01 17:29
dependencies:
- TASK-38493
assignee:
- '@codex'
labels:
- console
- agents
- planning
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md
- backlog/decisions/285-native-phased-agent-workflows.md
updated_date: 2026-11-01 17:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Translate the approved shared-worktree workflow design into ordered, independently reviewable implementation work that reuses existing Console execution and persistence owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The implementation plan covers every approved contract and all twelve acceptance scenarios with exact code/test owners and coherent interfaces.
- [x] #2 Atomic To Do implementation tasks exist in dependency order and link the accepted spec, ADR and plan without premature task implementation plans.
- [x] #3 The plan includes prerequisite qualification, meaningful red/green checks, recovery/backup/schema obligations and a real native provider qualification gate.
- [x] #4 Document links, source references, task metadata, task ID uniqueness, example syntax and scoped whitespace pass; product implementation is not started.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Direct planning of the accepted design; any change to those contracts requires an ADR/spec amendment before execution.

1. Finalize and commit only the approved design artifacts, preserving unrelated staged and working-tree changes.
2. Map actual code/test owners, define consistent interfaces and write a checkbox-based implementation plan with executable test examples.
3. Create ten atomic To Do tasks in dependency order, each referencing the accepted design/ADR and its detailed plan section; do not add their per-task Implementation Plans before work starts.
4. Self-review spec and A1–A12 coverage, interface/type consistency, example syntax, local links, metadata/ID uniqueness and whitespace.
5. Complete planning notes/status, commit only planning artifacts and present execution options without starting product implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the implementation plan for accepted ADR-285 and the finalized shared-worktree spec, plus ten atomic To Do implementation tasks in dependency order. The plan maps exact existing/new code owners, coherent authority/lifecycle interfaces, targeted red/green checks, bounded result storage, native worktree coordination, active-time accounting, recovery/backup/catalog obligations, Console UI and live qualification. A1–A12 each map to implementation and qualification tasks. Child-launch custody depends on task-scope enforcement; ordinary parent/child work shares activity reservations to avoid self-deadlock.

Self-review and documentation validation passed: 57 local links/anchors, eleven Python example syntax checks, task frontmatter/owned sections, dependency order, scoped task-ID uniqueness, twelve acceptance scenarios and whitespace. Existing source anchors and CommandParse fields were checked. The original staged index was confirmed unchanged after the design-only commit. No product code, migrations, provider execution or runtime tests were run. The historical raw_source_selection_changed prerequisite remains explicit rather than claimed fixed.

ADR required: no new ADR; backlog/decisions/285-native-phased-agent-workflows.md is the accepted contract linked from spec, plan and implementation records. Product execution remains unstarted.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
