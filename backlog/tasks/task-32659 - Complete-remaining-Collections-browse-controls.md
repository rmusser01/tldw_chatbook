---
id: TASK-32659
title: Complete remaining Collections browse controls
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 22:00'
updated_date: '2026-09-16 00:39'
labels:
  - library
  - collections
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up the Collections reader review with the remaining control-path findings: Clear preserves the active text search, More saved searches has no dispatch handler, and repeated Archive can replace the original Undo receipt. Confirm each through its user journey before repairing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clear removes the search and form filters responsible for the empty-result state, with valid sorting and truthful results.
- [x] #2 More saved searches reaches additional real saved searches and maintains active authority and scope.
- [x] #3 Repeated Archive cannot overwrite the original reversible status or present an action that silently changes its Undo meaning.
- [x] #4 Targeted and native evidence qualify the repaired controls at wide and compact sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md; backlog/decisions/055-library-destructive-action-reversibility-rule.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Complete the existing bounded saved-search continuation and repair filtering/Archive transitions within the current capture authority, paging, storage and token contracts.

1. Add real-service production-CSS journeys reproducing Clear leaving text search (including relevance sort), More saved searches failing to reach a second page, and repeated Archive changing the original Undo status. Cover 170x48 and 80x24 in both themes.
2. Make Clear reset all filter fields and relevance to a valid non-query sort while retaining scope predicates. Reuse the existing page-and-detail request path where equivalent.
3. Page the existing saved-search rail in bounded 20-row windows with More and Previous controls, a visible page range, retained active capture scope, explicit failure/retry, and authority/request fencing. Preserve the last good window on failure and return focus to a visible control after recomposition.
4. Disable Archive for an already archived capture with a readable reason; prevent the same transition at the scope-service boundary before revision or Undo state changes. Verify the original Undo status remains recoverable.
5. Run affected browse/reader/controller/service, layout/wiring and token/bundle/size checks; compare Ruff against base without raising budgets. Complete private native confirmation, exact read-only persistence and normal shutdown checks.
6. Update guide, QA and workflow audit, review the bounded diff, finish Backlog and commit locally. No full suite, provider/server/extraction calls, push or integration into dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed Clear/search recovery, bounded saved-search continuation and repeat-Archive protection. Clear resets text/domain/tags/dates and relevance sorting while retaining scope; expanded compact forms scroll and Apply/Clear occupy separate rows. Saved-search loading has explicit retry, last-good-window retention and authority/request fencing in a 49-line module; rail recomposition preserves surviving focus and falls back when a row disappears. Archived captures display disabled Archived, and the service rejects repeat Archive before revision/Undo changes.

Existing ADRs apply; no new ADR required: backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md, 055-library-destructive-action-reversibility-rule.md, 086-library-adaptive-reader-shell.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md. Reused equivalent capture request transitions and generalized deferred visibility repair without changing authority, schema, dependencies or token values.

Validation: 118 distinct targeted checks pass, including 20 new browse/loading checks. The unchanged LibraryScreen size check remains an inherited failure (35,210 lines / 1,320 methods versus 33,204 / 1,276). Collections budgets pass at 1,687/699/49; no existing budget raised. Zero new Ruff diagnostics, new-file formatting and changed-range formatting pass. Independent final review found no remaining finding in scope.

Native TldwCli/LinuxDriver run-002 passed 170x48 dark and 80x24 light. Six SVGs rendered/inspected; read-only SQLite confirmed exact capture statuses/notes, 21 unchanged saved searches, ten integrity checks and zero messages. Normal Quit returned exit 0; owned terminal closed. QA evidence: Docs/superpowers/qa/2026-09-15-collections-browse/README.md. Updated Collections guide and Library workflow audit. Wider compact Work toolbar clipping and manual text-search clearing under relevance remain for the next review. No full suite, provider/server/extraction calls, push or dev integration.
<!-- SECTION:NOTES:END -->
