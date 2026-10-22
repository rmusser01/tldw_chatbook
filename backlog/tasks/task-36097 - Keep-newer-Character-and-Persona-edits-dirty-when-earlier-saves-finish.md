---
id: TASK-36097
title: Keep newer Character and Persona edits dirty when earlier saves finish
status: To Do
created_date: 2026-10-22 01:24
labels:
- personas
- textual-patterns
- drafts
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-21-textual-book-after-pr3199.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An ordinary Character or Persona Save must acknowledge only the values it persisted. Typing made while that save is pending must remain an unsaved edit, so existing New, navigation and quit protections cannot silently discard it.

### Evidence and scope

Medium priority, audit order 1 of 4. Relevant source blobs on fetched merged dev `2a5afa03b31f60e00bdcd10a161e3e351fad023b` are identical to diagnostic candidate `8ff5f0e2d4f3016445545e84f9e30a5233b8fddd`. Mounted keyboard-driven Character and Persona witnesses held the existing real persistence entry, then delegated to actual SQLite or the local Persona JSON store. After a `-new` suffix was typed, persistence contained only the submitted value, but save completion marked the newer form clean. The aggregate navigation guard allowed leaving without a prompt, and actual New discarded the later text. Create paths are demonstrated; update shares the finisher but was not separately exercised.

Source: [tldw_chatbook/UI/Screens/personas_screen.py:15709](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/Screens/personas_screen.py#L15709); [tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py:773](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py#L773); [tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py:403](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py#L403).

Book basis: Chapter 7 printed 149 / PDF 158 and chapter 17 printed 326 / PDF 335 explain asynchronous handlers and worker-to-UI completion. Snapshot-aware save acknowledgement follows the observed defect and existing application contracts; the book does not prescribe a new persistence policy.

These are audit findings, not implemented fixes. The report distinguishes demonstrated cases from future regression controls.

### Existing ownership and decisions

TASK-481 Done covers save-in-place and sequential version preservation; TASK-33622.14 Done supplies the aggregate quit guard. This finisher incorrectly clears the predicate those guards use. TASK-33791 To Do covers selected Persona summaries, TASK-33910.8 To Do covers guarded transitions/load failures, and TASK-33910.14 To Do covers lore/dictionary entry and Options drafts. None owns this ordinary pending-save acknowledgement outcome.

Related tasks: [TASK-481](task-481%20-%20P3b-follow-up-sweep-deferred-editor-polish-minors.md); [TASK-33622.14](task-33622.14%20-%20Roleplay-CtrlQ-quits-past-unsaved-Roleplay-drafts-without-the-Save-Discard-Stay-guard.md); [TASK-33791](task-33791%20-%20Roleplay-persona-card-and-Inspector-tool-policy-line-show-a-different-persona-than-the-one-selected.md); [TASK-33910.8](task-33910.8%20-%20Roleplay-frame-B5b-1-one-guard-dialog-per-domain-executors-and-the-load-fence.md); [TASK-33910.14](task-33910.14%20-%20Roleplay-frame-B9a-entry-and-Options-draft-domains-no-layout-change.md).

ADR required: no new ADR for this bounded repair.
ADR paths: [046-roleplay-chat-display-identity-and-template-provenance](../decisions/046-roleplay-chat-display-identity-and-template-provenance.md); [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md).
Reason: No new ADR is required for a bounded correction within the existing editor owner. Accepted ADR-046 and its aggregate-draft amendments govern the behavior; application-session-state ADR-033 supplies ownership context. A new draft store, service contract or cross-screen mutation-lifetime scheme would require separate ADR review.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-21-textual-book-after-pr3199.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Completing a Character Save retains later form edits as unsaved while adopting the actual persisted record identity and version; the newer text is neither overwritten nor falsely acknowledged.
- [ ] #2 Completing a Persona Save provides the same guarantee for later edits and its actual persisted identity/version.
- [ ] #3 After either completion, existing New and leave/quit guards still detect the later edits; choosing Stay retains them, and no transition silently treats them as saved.
- [ ] #4 A subsequent explicit Save persists the remaining edits to the same record with the correct version; a Save with no later edits becomes clean and failed saves preserve drafts. Mounted real-store regressions distinguish these outcomes.
<!-- AC:END -->
