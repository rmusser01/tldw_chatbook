---
id: TASK-31245
title: Add Character chats mode to CtrlK switcher
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 02:09'
updated_date: '2026-09-06 17:39'
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
- [ ] #11 Library recovery closes the switcher only after the exact requested inspection is admitted; rejection or precommit cancellation preserves its query and highlighted identity.
- [ ] #12 Character Keyword pages preserve the repository relevance order and rejected query edits retain the prior accepted search without exposing stale actionable targets.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task5 delivery on frozen dev5894 preserves the separate switcher PR, current Keyword/live-authority contracts and typed Console activation. UI findings I1-I3 fixed at a83 and independently re-reviewed. User approved I4 on 2026-09-06: implement Library-owned display-neutral prepare and synchronous exact route/reader-selection commit before app overlay dismissal, with existing bounded local locator/save guards and request-owned precommit cancellation. Existing app navigation remains sole coordinator; cold entry consumes prepared selection without second admission race. Detailed steps are in Task5 of Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md. ADR required: yes, amendment to backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md (Task5 admission correction), linked spec amended. No new origin/repair authority, Meaning/runtime/schema, cap/dependency change, global rollback rewrite, or native loop. Focused RED/GREEN followed by one bounded affected gate, static/startup deltas as relevant, exact scoped review/commit. Keep In Progress and all unmet evidence explicit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task5 Keyword-only delivery integrated on frozen dev 5894f4755 in codex/task-31245-character-switcher. Scoped replay of 968680200,45c1f378e,04e04288d and applicable a1fd9c34a/b09c7af2fc/5359324c6 hunks adds Character chats, independent query ownership, typed activation, Context handoff and production-CSS compact layout. Existing ADR120/085/031/083 govern; no new ADR, Meaning, dependency or cap change. New Character-specific shared validation preserves raw 200-character/control boundary without narrowing incumbent Active/History or Context.search. Library completion waits for app navigation and retains the existing Context Character return anchor, but exact Library inspection admission remains unresolved pending scope choice; late global overlay-teardown failure remains inherited. One owning aggregate:381 passed7 failed (six exact baseline dismissal failures plus stale owned 512 expectation); the owned expectation and bounded painted/containment regressions passed subsequent focused checks, no aggregate repeat. Current startup635/660 imports,972/972 UI-ready,499/500 preimport modules,364400/378740 preimport LOC; final CSS796086/804000. No new Ruff diagnostic deltas; inherited size gates remain failing (ChatScreen whole-file +104 lines). Two synthetic production-CSS capture batches only; recovery-row wrapping and absent native/equal-terminal/scale evidence remain open. Report and complete log/capture inventory: .superpowers/sdd/2026-09-05-character-keyword-release-isolation/task-5-delivery-report.md. Keep In Progress and acceptance unchecked for independent review; no blanket Done claim.

Review fix round1 (I1-I3 only, reviewed BASE91eb8c3cd): pending Character search now owns the visible divider and clears stale selected detail/action hints; a redundant initial settled-query Input event no longer starts another search. Frozen activation identity is retained in flight, while failed recovery ownership is retired on row/query/mode transition and selected detail/Help follow the highlight. Character labels use mounted Button content width minus Textual line padding, preserving long unavailable/Unicode metadata on the second physical row. Focused RED9failed; final scoped GREEN9passed; one covering run59passed2failed (initial-query error cases), followed by exact correction3passed. No covering/aggregate/startup repeat. Three changed Python files pass Ruff/format; reviewed/full BASE whitespace pass; CSS unchanged. One review-directed synthetic12-state compact/wide SVG+text batch retained with existing bootstrap/native/font limitations. I4 exact Library admission remains open and unwaived; task remains In Progress for scoped re-review. Full appended evidence in task-5-delivery-report.md.

I4 correction against reviewed a83fd7fe: implemented ADR120-approved display-neutral Library prepare/synchronous exact selection commit, existing app FIFO pre-overlay admission controls, and modal-owned cancellable recovery task. New first-use Library helper205 lines; existing unavailable811/navigation controller198 unchanged, no cap raises. Frozen-source focused proof23passed (21 production-owner/SQLite I4 cases plus import/preimport budgets); static no added Ruff diagnostics. Bounded covering79passed3failed, then exact-three baseline/current comparison reproduced FIFO initial-screen and incomplete WorkspaceProbe failures on both; browse-size2 passed both in comparison, so earlier covering-only clipping remains unclassified. FD growth warnings remain unexplained. See packet task-5-delivery-report.md I4 receipt and raw /tmp/task31245-i4-*.log. In Progress/unchecked pending independent I4 review; inherited post-overlay synchronous stack-failure limitation remains.

Latest-dev integration: rebased all three reviewed Task5 commits from5894f4755 onto exact fetched devc4d45c0926580a8756cfa13c5463b1d0fc808c1a, without conflicts or source edits. Safety ref codex/task-31245-reviewed-pre-dev-rebase preserves1f18a2b317b64ed4a95bc26fe753a852b3ae02e2; range-diff marks all three patches equivalent, rebased source6e8a2ea76e108b296376503d8f36e7f5d6a92bfe. Upstream Canvas active-path, Library failure/retry, speaker rename and generated CSS retained. One bounded affected/four-budget gate79passed11warnings123.46s; CSS reproduces, no added Ruff diagnostics, whitespace clean. Budgets635/660imports,972/972UI-ready,499/500preimport,365925/378740LOC,Library110186/123319LOC,797010/804000CSSbytes. Narrow browse-size2 passed; earlier covering-only clip remains unresolved historical evidence. FD+282 warning unknown cause, inherited late post-overlay stack-failure qualification retained. No caps/dependencies/origins/Meaning changed, no remote/native actions/full sweep. See packet task-5-latest-dev-report.md and /tmp/task31245-latest-dev-*.log; remains In Progress/unchecked for controller final review/publication.

Final review fix wave against2184109b: corrected both confirmed findings in modal only. Live Active projection/receipt cache continues updating while frozen Character activation/recovery prevents result refresh (including queued workers) from invalidating its request generation; authority-change polling still closes fail-closed. Selected detail truncates to actual content cells and recomputes through existing settled mount/resize label synchronization; no CSS dimensions/caps changed. Focused RED reproduced4held-locator projection failures and2wide painted timestamp failures; fixture timezone assumption corrected separately. Focused GREEN8passed; frozen covering86passed2warnings82.02s across I4/switcher/geometry plus7named Active live selectors, including full absolute timestamp paint, ASCII/Unicode compact-wide-resize, accepted/cancelled/authority-changed real Library owners. Changed3Python files Ruff/format/whitespace clean; modal+9lines/no new methods. FD growth+412(12→424,limit200) remains unexplained; Requests mismatch and all previous acceptance/resource/native/performance qualifications retained. No startup/CSS/native/full sweep/dependency/cap/remote changes. See packet task-5-delivery-report.md final-fix receipt and /tmp/task31245-final-fix-*.log. Remains In Progress/unchecked for scoped controller re-review and user qualification decision.

Qodo current-head triage: preserve incumbent Context search raw512 and explicit switcher-handoff raw200 boundary (Keyword storage does not call the raw200 unavailable-page validator). Correct ranked-page projection, rejected-edit ownership, and introduced public docs/types. Destination-only rollback for a transfer that never owns Library awaits a separate user scope decision; no global stack rollback or implicit qualification waiver.

Qodo comments2-5 correction against16072cbf6: removed redundant Character projection recency sort so repository relevance/page order survives stable deduplication; rejected Character edits restore the prior accepted raw query before changing committed page/target or pending search ownership. Added concise introduced public API docs/types with TYPE_CHECKING-only Library references. Context512 and switcher/handoff200 remain unchanged; comment6 retained-Library contamination awaits separate scope approval, no lifecycle/rollback edits. Focused RED7failed; final GREEN7passed after bounded recovery-copy paint correction. One affected gate110passed4warnings69.82s; post-gate type-only import ordering correction verified by final static/no-new-diagnostics and two import tests2passed3warnings3.52s. Imports635/660; preimport499/500,365941/378740LOC,Library110202/123319. FD growth+412 and Requests mismatch remain unexplained/unwaived; no full sweep/native/remote/rebase/cap changes. Controller plan amendment and AC12 preserved, task stays In Progress/unchecked. Full commands and stopped-process receipt in packet task-5-delivery-report.md Qodo section and /tmp/task31245-qodo-*.log.
<!-- SECTION:NOTES:END -->
