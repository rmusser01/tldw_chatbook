---
id: TASK-31928
title: Repair Console Stop clipping after Redirect action was added
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 06:18'
updated_date: '2026-09-06 15:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The real 160x48 Console action row budgets 37 cells while active controls require 47 after TASK28227 introduced Redirect. Stop is clipped by hidden overflow. Forced test scrolling conceals the production defect; preserve the original regression until a bounded runtime layout fix is approved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At the existing 160x48 viewport the active Stop button is physically visible and clickable without forced scrolling or programmatic Stop focus.
- [x] #2 The action-width calculation accounts for applicable controls without changing Redirect semantics or widening test deadlines.
- [x] #3 Real click/cancellation and relevant composer layout tests pass, with ordinary composer focus established for synthetic Send setup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow approved-console-regressions plan TASK31928: RED physical Stop containment with ordinary composer focus, include Redirect in existing width budget, GREEN real click and complete layout/attachment tests, independent review. ADR required: no; routine layout correction preserves control semantics and DESIGN.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved bounded layout repair, refined after 80-column baseline: reserve Redirect 10 cells only while active; reuse existing deferred draft reflow on run-state changes so Stop does not move between mouse-down and mouse-up; let optional attachment labels yield narrow-row space while retaining the full tooltip. Existing order, visibility, 160x48 viewport and 0.5-second click deadline remain unchanged. Original containment RED; final actual Stop plus width controls14pass, complete attachment/width35pass with no native retained SQLite, complete command104pass, streaming/width102pass, overflow/narrow/cursor84pass1baseline Retry Speech failure reproduced with original methods restored. Logs /private/tmp/tldw-31822-{final-green,attachments-width-full,command-full,responsive-green,layout-full,retry-speech-baseline}.{xml,log}. Fixture retention in adjacent command/dictation files is TASK31927, not waived. Independent code/spec review, scoped lint/format/diff checks and Impeccable layout scan pass. No new ADR; existing dense inline action row preserved.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31822 was renumbered to TASK-31928 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
