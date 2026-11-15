---
id: TASK-37360
title: Preserve Media filter typing and editing context across browse publication
status: Done
assignee:
  - '@codex'
created_date: 2026-10-29 04:49
updated_date: '2026-11-15T22:29:00Z'
labels:
  - textual-patterns
  - library
  - media
  - inputs
dependencies: []
references:
  - tldw_chatbook/Widgets/Library/library_media_canvas.py
  - tldw_chatbook/UI/Library_Modules/library_media_controller.py
documentation:
  - Docs/superpowers/qa/2026-11-14-textual-pattern-followups/verification.md
  - Docs/superpowers/plans/2026-11-14-textual-pattern-followups.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep ordinary typing and editing in the Library Media filter intact while loading and accepted browse results publish. A user pausing between letters currently loses the previously typed prefix, so the visible results no longer reflect the query they were composing.

**Evidence:** PR3290's unchanged Reader-identity control failed on source3e414ff824 with applied query `t` and two rows after native `roduct` keys. A native observation with the same keys spaced1.2seconds apart reproduced that exact terminal state. Loading/results create fresh filter Inputs; restored focus selects their prefill, and the next printable character replaces it. The trace shows that character reaching the current attached field, so a stale retained widget is not required. The uninstrumented CI interleaving is inferred from this sufficient matching witness; natural frequency and latency are unmeasured. One native Paste prepares the separate Reader-identity test successfully but does not repair typed input. [Investigation and raw-receipt map](../../Docs/superpowers/qa/2026-10-28-textual-followups-implementation/ci-repairs/media-filter-investigation.md).

**Existing owners:** [TASK-32350](<task-32350 - Library-Media-list-the-filter-box-and-the-applied-filter-can-disagree-—-add-a-persistent-scope-line.md>) owns truthful applied-scope presentation, not preservation of this typed draft. [TASK-28013](<task-28013 - Library-media-list-in-canvas-filter-input-and-sort-control.md>) introduced the in-canvas filter/sort. [TASK-15457](<task-15457 - Library---convert-per-click-whole-screen-recomposes-to-canvas-scoped-sync.md>) and [TASK-281](<task-281 - Library-targeted-updates-instead-of-whole-screen-recompose.md>) concern broader screen-to-canvas update costs. [TASK-34230](<task-34230 - Media-focus-guard-is-released-before-the-recompose-it-covers-when-canvas-sync-falls-back-to-a-whole-screen-recompose.md>) concerns an unmounted-canvas whole-screen fallback guard; this witness uses an attached live Items field. [TASK-34000.18](<task-34000.18 - Library-Media-Read-later-trash-and-Edit-metadata-act-on-the-item-opened-by-link-not-row-1.md>) owns Reader action identity and is preserved by the native Paste setup. [TASK-34000.37](<task-34000.37 - Library-keyboard-focus-survives-dialogs-and-filter-submits-F6-reaches-every-canvas-F1.md>) covers other keyboard/dialog/filter-submit journeys. Coordinate those boundaries rather than duplicating their completed or broader scopes. The fresh417-ref/368-tree/55-worktree task screen found no discovered task acceptance scope for this exact ordinary-debounce prefix replacement.

**ADR assessment:** New ADR expected:no; ADR path:N/A. Existing [ADR-033](../decisions/033-application-session-state-ownership.md), [ADR-086](../decisions/086-library-adaptive-reader-shell.md), [ADR-150](../decisions/150-design-token-system-and-design-language.md) and [ADR-161](../decisions/161-component-pattern-library.md) preserve destination ownership, adaptive layout and design patterns. Reassess only if implementation changes those boundaries. No new query engine, storage contract or remote fallback is requested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native ordinary and paced typing across loading and accepted Media browse publications retains the complete current filter draft and its caret/selection/focus instead of replacing earlier characters.
- [x] #2 An earlier browse completion cannot overwrite later accepted filter edits or editing context; the scope line continues to describe the actually applied results until newer results are accepted.
- [x] #3 Clear/type/sort/paging behavior and dirty Reader off-page identity/highlight/action contracts retain their established outcomes.
- [x] #4 Mounted native controls expose the original prefix-replacement mechanism and verify the repaired publication boundaries, with all holds drained inside the App lifetime and unchanged test/privacy guards.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: repair existing retained-control behavior under ADR-033/086/161 without changing ownership or public contracts.
1. Reproduce paced native Media typing with exact query, identity, caret/selection/focus assertions before the production change.
2. Retain filter/chrome, recompose only results, and move post-recompose callbacks with their owner; preserve newer live drafts.
3. Repair the clean-Reader zero-results whole-screen replacement using existing targeted sync.
4. Verify held completion, Clear, truthful applied scope, dirty Reader, sort/pager and scoped sync with targeted tests and relevant style/governance checks.
Plan: Docs/superpowers/plans/2026-11-14-textual-pattern-followups.md

PR #3569 integration with dev cc6006fcf5: preserve the retained Media filter while combining incoming empty-page focus recovery. Review found a second deferred callback can outlive a newer focus intent; reproduce that ordering with the actual helper, keep the existing route/foreign-focus/pruning/eligibility safeguards, and enforce ownership at the final focus boundary. Run the existing Media preservation and incoming recovery checks plus a focused held-callback regression before publication. ADR required: no. ADR path: existing ADR-033/086/161 and current Library ownership contracts. Reason: routine focus-preservation repair at the existing asynchronous completion boundary, not a new navigation model.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retained the native Media filter and moved replaceable rows/toolbar/pager and their callbacks into the results owner. Synchronous edit admission preserves raw text, focus, caret and selection before queued Changed delivery. Scope chrome owns its pending display snapshot; clean zero-result Reader updates no longer replace the screen.

Changed the Media canvas, canvas_sync callback routing, clean Reader path and source/generated Library CSS; added mounted regressions and updated Library architecture/testing lessons. No controller/debounce, persistence or privacy-guard change.

Verification: original identity/prefix and before-Changed failures reproduced. Final 7 cases pass (5 new + 2 applied-scope regressions); broader targeted run had 52 passes. Four initially bootstrap-blocked legacy cases subsequently passed under the existing bootstrap contract. One early-Clear timing assertion independently fails on the exact baseline; settled native Clear passes. No assertions or guards weakened. Relevant CSS/token/component checks pass; no new lint findings; inherited whole-file formatting/lint debt is documented.

Independent task/scoped and final whole-branch reviews approved. Five new cases are in the PR census. ADR required: no; existing ADR-033/086/150/161 apply. Evidence and limitations: Docs/superpowers/qa/2026-11-14-textual-pattern-followups/verification.md. Local uncommitted implementation; no full-suite or publish claim.

PR3569 integration with dev cc6006fc preserves incoming empty-page recovery through retained results. A targetless reset no longer mistakes Textual automatic filter parking for an explicit filter request; native filter requests retain priority. Recovery captures accepted result, entry generation and focus, then checks canonical Media Find state and ownership again at the final focus write. Existing route, foreign-focus, pruning and eligibility safeguards remain. The final real-mounted regression loops12scenarios without increasing the722unbound-call pin. Its negative control catches the stale final write; original compact2 and final Media7 all pass. Exact line/no-slack and coupling guards pass; after explicit per-iteration callback binding, the final new selector passes with full Ruff/scoped format clean. Final production inventory has no drift and independent integration review found no actionable issue. No new production lint findings:24Library findings match HEAD exactly. ADR required:no new ADR; existing ADR033/086/150/161 govern this routine integration. Final source hashes and receipts, including initial failures and probe limitations: Docs/superpowers/qa/2026-11-14-textual-pattern-followups/ci-resync-cc6006fc.md. CI capacity follow-up TASK39144 and broader Settings parent remain open; no full local suite or final remote-CI claim.
<!-- SECTION:NOTES:END -->
