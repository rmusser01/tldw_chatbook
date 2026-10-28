---
id: TASK-37034
title: Scope Speech playground footer actions to the active view
status: In Progress
created_date: 2026-10-27 18:01
labels:
- textual-patterns
- speech
- usability
priority: low
references:
- tldw_chatbook/UI/Screens/stts_screen.py
- tldw_chatbook/Widgets/AppFooterStatus.py
- Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251.md
assignee:
- '@codex'
updated_date: 2026-10-27 23:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Speech footer hints truthful when the active view has no Playground. After selecting Audio Effects, the footer advertises Generate, Random text, Clear, Play and Stop even though their Playground target is absent and the keys silently do nothing.

A mounted production-CSS observation first proves Clear works from the Playground rail, then activates the real Effects rail button with Enter. The view worker removes Playground, but the rendered custom-footer text and active mirror bindings still advertise all five actions. No audio/provider service was invoked. Only Effects was dynamically qualified as the adverse view.

**Scope and related owners:** [TASK-2951](<task-2951 - task-1266-AC4-is-false-on-dev-TTSPlaygroundWidget-was-restored-and-never-re-deleted.md>) removed the dead legacy Playground and repointed mirrors to the live pane while it is mounted. [TASK-2610](<task-2610 - Lab Speech mode crashes the app with DuplicateIds on mount.md>) fixes duplicate mount IDs. [TASK-35869](<task-35869 - Make-Evals-grid-footer-and-F1-help-follow-active-keyboard-scope.md>) is the completed Evals-grid footer analogue, and [TASK-36694](<task-36694 - Handle-each-Speech-Recognition-button-activation-once.md>) owns Recognition button duplication. None owns current Speech view hint scope. The custom AppFooterStatus projection must be checked; default Textual Footer/check_action behavior alone does not establish it.

**ADR assessment:** No new ADR is expected for a bounded repair preserving existing ownership and service contracts. Existing [ADR-031](<../decisions/031-tui-keybinding-and-footer-hint-conventions.md>), [ADR-011](<../decisions/011-chatbook-workbench-ui-system.md>), [ADR-150](<../decisions/150-design-token-system-and-design-language.md>), [ADR-161](<../decisions/161-component-pattern-library.md>) apply; reassess if implementation changes those boundaries.

Evidence, source pins, book pages, controls and limitations: [post-merge Textual audit](../../Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251.md). Audited merged dev: `416feb50ee667fdc95cb269bf50f8b1d5b3ff8ff`. The supplied book is reference data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the active Speech view has no Playground target, the custom footer does not advertise its Generate, Random text, Clear, Play or Stop no-op actions; applicable mode and global hints remain available.
- [x] #2 Returning from Effects to Playground restores the applicable Playground hints and the existing working rail Clear action.
- [x] #3 Preserve printable input routing, current mode/global shortcuts and reserved-key conventions; computing action availability starts no audio, recording, playback or provider work.
- [x] #4 Mounted tests drive the actual Effects rail via Enter and the Playground return, assert rendered custom-footer text and actual action behavior, and include the ordinary Playground Clear control without audio services.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce production-CSS Playground rail Clear, Effects Enter, rendered custom-footer text, native binding scope, return and printable input without audio. 2. Add one typed local content-change message in existing STTS_Window.py after actual removal and successful ready replacement; owning STTSScreen rejects retired windows and derives footer/native availability from current view plus actual live Playground. Preserve MRO mount hooks. Verify held native return mount leaves hints absent until target exists. 3. Run targeted navigation/mount/button/shortcut and unchanged key/source/boot guards; inspect actual parent/private-child XML and source identity. 4. Update speech architecture and evidence; keep In Progress for root integration. ADR required: no. ADR path: N/A. Reason: routine bounded bugfix directly implements existing explicit widget-message ownership in backlog/decisions/011-chatbook-workbench-ui-system.md and active-context footer truthfulness in backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; preserves backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md. No new provider/runtime/persistence boundary; root approved existing STTS_Window.py scope for the local notification.
Root-authorized same-AC recomposition follow-up: revoke retired STTSWindow ownership and clear cached/rendered hints to the Lab set before native screen recomposition, then restore only after the new owned target is ready. Add a separate held native recomposition/old-owner mounted scenario under the unchanged 180s test budget; preserve existing action assertions and target readiness. ADR required: no; ADR path: N/A; direct existing widget-message/context ownership under backlog/decisions/011-chatbook-workbench-ui-system.md and backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md still apply, no new service/runtime or persistent contract.
<!-- SECTION:PLAN:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented target-backed Speech Playground mirror availability and custom footer registration in UI/Screens/stts_screen.py. The existing STTS_Window.py now emits one typed owner-tagged content-change notification after actual removal/replacement readiness; retired owners cannot restore stale hints. Screen recomposition revokes the old owner and clears cached/rendered hints before native removal, while a refused native swap restores only its still-attached owner. Existing save_state/current-view/axis semantics remain unchanged. Docs/Architecture/speech.md documents ownership. ADR required: no; direct existing backlog/decisions/011-chatbook-workbench-ui-system.md and backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md contracts, preserving backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md.
Tests/UI/test_speech_footer_scope.py proves actual production-CSS rail Clear, Effects Enter, absent/restored custom-footer child text and native bindings, held return mount, printable input, held native screen recomposition and old-owner rejection. Both original behavior and recomposition gaps produced clean RED parent/private-child assertions. Final task4-final02 has 9 parent passes and 2 actual private-child passes, no errors/skips; all relevant old Speech behavior controls qualified across scoped runs. Exact native profile-guard chains justified supported bootstrap_profile opt-ins in four legacy modules. Native mount/Studio worker settlement preserves their substantive assertions and budgets. The stale snapshot expectation was corrected to fixture-consistent 4/5 plus explicit OmniVoice-missing inspector evidence; capability product files are unchanged.
Fresh boot count 680/686 and preimport 558/558 modules, 424406/425347 LOC, maximum route 130690/135111 LOC pass unchanged limits. Focused Ruff passes; seven inherited E721 findings and two existing formatting debts are source-qualified. Broad unchanged MRO scanner reports five pre-existing sites; unchanged 8-second import-time guard fails with both candidate (15.1088s) and exact original Speech files (16.3614s), with native bytes restored in finally. Five prior Windows visual guard failures remain explicitly source-qualified with 87 unchanged inputs; no styles/tokens changed or allowances raised. Full evidence and raw/XML identities: .superpowers/sdd/2026-10-27-textual-37031-37034/task-4-report.md. Status remains In Progress for root review, integration, current-head CI and final AC closure.
Independent Superpowers task review approved: spec compliant, zero Critical/Important findings; final 9 parent cases and 2 actual private children passed with zero errors/skips. Root is completing rebased combined verification, whole-branch review and current-head CI before final task closure. Existing lint/format/MRO/native timing qualifications remain explicit; no allowances changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->