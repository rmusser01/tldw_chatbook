---
id: TASK-32701
title: Review Library Conversations filter and reader continuity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 13:35'
updated_date: '2026-09-16 14:03'
labels:
  - ui
  - conversations
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review the next Library destination while real Import execution remains blocked by host worker allocation, preserving the established conversation browsing and reader contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 List filtering and empty-result recovery retain an editable, visibly focused control at compact and wide sizes.
- [x] #2 Opening a conversation and finding text preserve the intended transcript and user editing/reading context through supported resize transitions.
- [x] #3 Any demonstrated consistency defects have focused regression coverage and use the existing design language.
- [x] #4 Bounded native checks in both themes use disposable conversations and a private profile, with targeted/static verification and documented limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: review and repair existing presentation/focus contracts without changing conversation authority, persistence or application structure.

1. Trace list filtering, retained reader synchronization and responsive pane/focus rules; inspect representative production styles and tests.
2. Reproduce browse/filter/find/resize journeys with production styles in a mounted harness and isolate demonstrated defects.
3. Add focused regressions before bounded fixes; preserve user focus, input selection and supported reader identity.
4. Run real TldwCli with private seeded conversation data, inspect compact/wide dark/light captures in bounded batches, verify clean exit and persistence.
5. Run targeted tests/static checks and independent review; update evidence and task before a local commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Library Conversations now owns its resize focus independently of Notes, preserves filter selection on explicit Items reopen, and keeps focused Find results visible through nested viewport scrolling. Delayed reveal work respects newer focus. Source CSS uses existing tokens; the generated Library sheet was rebuilt.

Validation:10 new production-style continuity cases plus2 adjusted F6/Escape checks pass;158 other selected neighboring checks passed (170 distinct passes). Two unchanged closeout failures reproduce on baseline: a media selector/ID assertion and hidden compact Notes Items. Original regressions fail on baseline; removing the focus guard fails both race/Find cases. New Python files are lint-clean;215 existing Ruff findings are unchanged and pre-existing formatting debt is recorded. No full suite ran.

Real TldwCli/private-database run004 passed compact/wide/short-height journeys in both themes. Ten captures were inspected once,10 private DBs passed integrity,2 conversations/26 message bodies remained exact,3 default-profile config hashes were unchanged, and normal exit/PID absence preceded closing the owned terminal. Independent review found no actionable defects and confirmed keyboard expectations.

Evidence and reproducible runner: Docs/superpowers/qa/2026-09-16-conversations-continuity/README.md. The nested-scroll incident is recorded in backlog/docs/lessons-testing-evidence.md. Changes are limited to Library screen/controller, conversation widgets/styles, focused tests and evidence. No new ADR: existing ADR-086, ADR-031, ADR-150 and ADR-161 apply. TASK-32700 remains independently blocked by host semaphore allocation; no Import success is claimed.
<!-- SECTION:NOTES:END -->
