---
id: TASK-32720
title: Review Search RAG recovery navigation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 15:44'
updated_date: '2026-09-17 16:07'
labels:
  - library
  - search-rag
  - verification
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Library component review so empty-source and provider recovery actions remain keyboard reachable and return users to a truthful Search/RAG state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty-library Import recovery is visibly keyboard reachable, opens the existing Import media canvas, and preserves the Search/RAG draft on return.
- [x] #2 Missing-provider and missing-credential recovery is visibly keyboard reachable, opens Providers and Models settings, and preserves the query, mode and source choices while refreshing readiness on return.
- [x] #3 Recovery navigation makes no unintended retrieval, answer or ingest calls; targeted tests, private native checks in both themes and sizes, static checks, review and evidence qualify the result.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: existing backlog/decisions/003-settings-library-rag-defaults.md, backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md, backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md. Reason: routine repairs to existing recovery, retained-screen refresh and query-mirroring contracts; no new application structure or service boundary.
1. Keyboard probes exposed the absent Settings action when no provider is selected; use the canonical existing provider blocker and Settings recovery pointer.
2. Whole-screen restoration probes exposed a blank rail query beside the restored panel draft; synchronize the sibling before the equal-state early return. Native retained-screen navigation exposed stale provider readiness after Settings; refresh the existing gate/status on resume while preserving results/history. Add focused regression cases before each repair.
3. Verify with targeted recovery/state/history/race/return tests and private native dark/light journeys at 170x48 and 80x24. Inspect captures, perform static checks and independent review, update guide/audit/task evidence and commit locally. No full suite, push or merge.
Allocation: reachable task paths across 305 refs and 27 worktrees had maximum 32719 before creation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Search/RAG now offers the existing Providers and Models action for both an unselected provider and a missing credential, mirrors restored drafts into both query fields, and refreshes Run/recovery when a retained Library screen resumes. The existing serialized refresh preserves results/history, and navigation never submits work. No new architecture, styles or tokens were introduced; existing ADR-003/031/150/161 apply.

Final targeted verification passes 324 tests in 293.58s, including 21 new keyboard/theme/size/navigation/return-state cases. Red evidence covers the absent provider action, blank restored rail query, and four retained-screen stale-gate cases. A preexisting credential test was updated to TASK-32236s established split: human Settings guidance on screen and technical remedy in the recovery record. New-file lint/format and changed-range formatting pass; existing large files have no new lint diagnostics.

The private native app passes twelve actual Import/Settings round trips in dark/light at 170x48 and 80x24, with eight inspected captures, zero retrieval/answer/ingest calls, unchanged source/default files, eleven healthy private databases, zero conversation messages, normal exit and exact PID absence. Two native setup failures targeted a hidden compact Import rail; using its existing Nav handle corrected the runner. Native provider changes are ephemeral config fixtures, not saved Settings edits or live server credential validation.

Independent review requested a stronger current results-container identity assertion and accurate pending-test wording; both are resolved, with no remaining findings. User guide, audit, testing lesson and QA evidence are updated at Docs/superpowers/qa/2026-09-17-rag-recovery-navigation/README.md. No full suite, push or merge. Next bounded review: Providers and Models keyboard editing, save/cancel and return feedback.
<!-- SECTION:NOTES:END -->
