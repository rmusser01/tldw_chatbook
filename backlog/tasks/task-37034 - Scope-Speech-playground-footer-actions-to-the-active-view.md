---
id: TASK-37034
title: Scope Speech playground footer actions to the active view
status: To Do
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
- [ ] #1 When the active Speech view has no Playground target, the custom footer does not advertise its Generate, Random text, Clear, Play or Stop no-op actions; applicable mode and global hints remain available.
- [ ] #2 Returning from Effects to Playground restores the applicable Playground hints and the existing working rail Clear action.
- [ ] #3 Preserve printable input routing, current mode/global shortcuts and reserved-key conventions; computing action availability starts no audio, recording, playback or provider work.
- [ ] #4 Mounted tests drive the actual Effects rail via Enter and the Playground return, assert rendered custom-footer text and actual action behavior, and include the ordinary Playground Clear control without audio services.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
