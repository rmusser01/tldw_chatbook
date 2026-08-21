---
id: TASK-19641
title: Build the returning-user Library landing
status: In Progress
assignee: []
created_date: '2026-08-21 21:49'
updated_date: '2026-08-21 22:51'
labels:
  - library
  - ux
  - landing
dependencies:
  - TASK-19022
references:
  - >-
    Docs/superpowers/specs/2026-08-20-library-lifecycle-progressive-disclosure-design.md
  - backlog/decisions/076-library-lifecycle-progressive-disclosure.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give populated Library profiles a truthful action-oriented landing that helps regular users resume work, recover current problems, and reach trustworthy cached summaries without synchronous scans, fabricated state, or power-user regression.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At the wide breakpoint, a populated returning-user landing presents Continue from the current user-admitted eligible Library route only when its full scope has authoritatively applied; late older successes cannot replace it.
- [ ] #2 A prior item/detail view resumes at its valid source-list scope with neutral pre-read copy and no fabricated selection; deletion, invalidity, or page clamping is disclosed only after the source owner’s authoritative read.
- [ ] #3 Needs attention appears only for a current screen-owned recoverable failed import or stale state, provides its existing Review or Retry path, and is not implied to survive restart.
- [ ] #4 From your Library uses only existing trustworthy cached summaries in fixed source order, omits unavailable or unresolved data without implying cross-source chronology, and performs no synchronous scan, ranking, or new source read during composition.
- [ ] #5 Quick actions reuse the existing Import, New note, and Search routes in that order with guarded semantic focus restoration and no duplicate router or action owner.
- [ ] #6 Compact presentation remains rail-first; wide and compact production-CSS tests prove Continue-first hierarchy, truthful F6/Tab order, semantic focus transfer, terminal-cell containment, and recovery at 170x48 and 100x30.
- [ ] #7 Only modified or touched landing components and direct owners are tested; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the retained Library landing state/canvas with optional Continue and Needs attention sections, present cached summaries as From your Library in fixed source order, and intentionally use Import → New note → Search as the quick-action order.
2. Capture a validated primitive Continue receipt synchronously at save_state() from the current user-admitted eligible route with authoritative applied state; store it separately from the active route through runtime-scoped ScreenStateStore. Support Conversations, Database Notes, Media, Prompts, Skills, Collections, and Search only; exclude File Notes, item/editor, Import/Export/Trash, and Study handoffs.
3. Dispatch Continue through existing guarded source routes. Restore Media/Prompt scope, Conversation page/query, and Database Notes sort/filter. Use neutral pre-read source-list copy; let source owners disclose deletion or clamping only after authoritative reads.
4. Derive one current-screen recoverable failed-import or stale attention item without persistence, and targeted-sync the retained landing on fail/requeue/dismiss through existing registry/source owners.
5. Prove From your Library uses only trustworthy cached summaries with no I/O or cross-source chronology claim, while existing Import/New note/Search routes and semantic focus ownership remain intact.
6. Verify Continue-first hierarchy, F6/Tab truth, compact semantic focus transfer, terminal-cell containment, runtime compatibility in ScreenStateStore, and production CSS at 170x48 and 100x30. Run only touched/direct-owner tests and exact Ruff/diff gates, update docs, review, and close through Backlog CLI.
<!-- SECTION:PLAN:END -->
