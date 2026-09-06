---
id: TASK-31245
title: Add Character chats mode to CtrlK switcher
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 02:09'
updated_date: '2026-09-06 15:23'
labels:
  - console
  - switcher
  - characters
  - ux
dependencies:
  - TASK-31244
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the Console session switcher into a complete operational switchboard for active tabs, history, and local character conversations while preserving all incumbent target-trust behavior.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-31237 on 2026-09-04. The final pre-commit worktree sweep
found the older `Reader uses its vertical space` task created at 01:50; it keeps
TASK-31237 under the older-arrival rule. This unshipped task moves with all plan
and dependency references.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ctrl+K exposes Active, History, and Character chats with F3 cycling and truthful visible hints.
- [ ] #2 Blank Active Enter still targets the most recently used other tab; explicit navigation and nonblank queries activate only the committed highlighted identity.
- [ ] #3 Active and History share their per-visit query and labeled zero-match widening; Character chats owns a separate query and never widens.
- [ ] #4 Character rows use the approved two-line grammar plus one stable selected-only detail region, with no unselected snippets.
- [ ] #5 The modal stays mounted through typed cancellable activation, freezes the committed row, ignores post-commit Escape, and cannot duplicate opens.
- [ ] #6 The exact 52x20 row budget, focus order, cell-aware truncation, paging, pointer press target, F2 restrictions, and Cancel reachability are enforced.
- [ ] #7 Context shows Continue search in Character chats only when this mode is available and transfers a validated query without pretending Meaning exists.
- [ ] #8 Targeted trust, modal dismissal, activity, keyboard, focus, geometry, zero-result, and exact-resume tests pass with production CSS.
- [ ] #9 Navigation and Keyword delivery is isolated on frozen dev with the original five task boundaries, no Meaning runtime or controls, and all applicable later correctness fixes.
- [ ] #10 Fresh targeted tests, startup comparison, resource ownership, static checks and bounded Pilot evidence are recorded with inherited failures and unavailable external evidence explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task5 delivery after PR2452 merge5894f4755 (2026-09-06): preserve the existing separate task boundary; replay only Task5 switcher commits968680200,45c1f378e,04e04288d and applicable later Keyword-only fixes onto current dev in codex/task-31245-character-switcher. Preserve merged Task2 snapshot/live-revision contracts and Task3/4 activation/return ownership. Record incumbent baseline, use focused RED/GREEN for new integration corrections, then targeted switcher/Context/activation tests, production-CSS compact/wide painted verification, startup and changed-file static checks. ADR required: no new ADR. ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md; ADR085/031/083 apply. Reason: direct implementation of approved third mode and Keyword delivery, no Meaning/runtime/schema/dependency expansion. Keep existing release-isolation AC and historical evidence; report new baseline limitations separately, no blanket Done or native verification claim.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task5 Keyword-only delivery integrated on frozen dev 5894f4755 in codex/task-31245-character-switcher. Scoped replay of 968680200,45c1f378e,04e04288d and applicable a1fd9c34a/b09c7af2fc/5359324c6 hunks adds Character chats, independent query ownership, typed activation, Context handoff and production-CSS compact layout. Existing ADR120/085/031/083 govern; no new ADR, Meaning, dependency or cap change. New Character-specific shared validation preserves raw 200-character/control boundary without narrowing incumbent Active/History or Context.search. Library completion waits for app navigation and retains the existing Context Character return anchor, but exact Library inspection admission remains unresolved pending scope choice; late global overlay-teardown failure remains inherited. One owning aggregate:381 passed7 failed (six exact baseline dismissal failures plus stale owned 512 expectation); the owned expectation and bounded painted/containment regressions passed subsequent focused checks, no aggregate repeat. Current startup635/660 imports,972/972 UI-ready,499/500 preimport modules,364400/378740 preimport LOC; final CSS796086/804000. No new Ruff diagnostic deltas; inherited size gates remain failing (ChatScreen whole-file +104 lines). Two synthetic production-CSS capture batches only; recovery-row wrapping and absent native/equal-terminal/scale evidence remain open. Report and complete log/capture inventory: .superpowers/sdd/2026-09-05-character-keyword-release-isolation/task-5-delivery-report.md. Keep In Progress and acceptance unchecked for independent review; no blanket Done claim.

Review fix round1 (I1-I3 only, reviewed BASE91eb8c3cd): pending Character search now owns the visible divider and clears stale selected detail/action hints; a redundant initial settled-query Input event no longer starts another search. Frozen activation identity is retained in flight, while failed recovery ownership is retired on row/query/mode transition and selected detail/Help follow the highlight. Character labels use mounted Button content width minus Textual line padding, preserving long unavailable/Unicode metadata on the second physical row. Focused RED9failed; final scoped GREEN9passed; one covering run59passed2failed (initial-query error cases), followed by exact correction3passed. No covering/aggregate/startup repeat. Three changed Python files pass Ruff/format; reviewed/full BASE whitespace pass; CSS unchanged. One review-directed synthetic12-state compact/wide SVG+text batch retained with existing bootstrap/native/font limitations. I4 exact Library admission remains open and unwaived; task remains In Progress for scoped re-review. Full appended evidence in task-5-delivery-report.md.
<!-- SECTION:NOTES:END -->
