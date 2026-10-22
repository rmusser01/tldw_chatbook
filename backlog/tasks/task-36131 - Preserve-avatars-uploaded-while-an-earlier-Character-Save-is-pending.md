---
id: TASK-36131
title: Preserve avatars uploaded while an earlier Character Save is pending
status: To Do
created_date: 2026-10-22 15:29
labels:
- personas
- textual-patterns
- drafts
- attachments
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-22-textual-book-after-pr3205.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Character Save must acknowledge only its submitted avatar. A later valid upload must remain staged and unsaved when that earlier Save completes.

### Evidence and scope

Medium priority; audit order 2 of 3. Audited source blobs on merged dev `7700c63a688355a2df595941b8b2b4dc8a819074` match diagnostic candidate `e41fc622e2702b3e87a1f21d83f1f328ed48b821` exactly. Mounted private-profile native test uses actual Ctrl+S, real Upload Button, fully mounted EnhancedFileOpen and its selected-path dismiss seam, actual validated file read/staging and original SQLite update delayed by a threading Event. Initial red PNG is persisted at version 1. During a held update that submitted red, selecting a different blue PNG stages blue with attachment_dirty=true and aggregate_dirty=true. Completing the original real update persists red at version 2 and changes the live editor back to red, attachment_dirty=false, form_dirty=false and aggregate clean. The ordinary upload-before-Save control persists blue at version 2 and becomes clean. Corrected run-2 has 1 passing control and 1 intended avatar-retention failure in 56.03 seconds. Initial run-1 premature picker dismissal caused secondary NoMatches before picker Mount completed and is excluded. No source changes, external backend calls or simulated storage success.

Source: [tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py:786](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py#L786); [tldw_chatbook/UI/Screens/personas_screen.py:10784](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/Screens/personas_screen.py#L10784); [tldw_chatbook/UI/Screens/personas_screen.py:15744](https://github.com/rmusser01/tldw_chatbook/blob/7700c63a688355a2df595941b8b2b4dc8a819074/tldw_chatbook/UI/Screens/personas_screen.py#L15744).

Book basis: Chapter 7 printed 149 / PDF 158: Asynchronous handlers and background work; the application must retain the correct user state while awaiting work. Chapter 17 printed 326 / PDF 335: Worker-to-UI completion and result applicability. The reference does not prescribe a save acknowledgement policy; the observed attachment owner controls this finding.

### Existing ownership and decisions

TASK-36097 Done covers raw text/greetings; its implementation QA explicitly identifies staged-avatar handling as outside the accepted scope at Docs/superpowers/qa/2026-10-21-textual-save-and-source-freshness.md:39. This is newly demonstrated attachment-domain loss, not another text/greetings task. TASK-100 Done restores ordinary upload/Save; TASK-481 Done covers sequential Save/version/thumbnail behavior. TASK-31243 In Progress and TASK-33910.8 To Do cover navigation/activation guards and discard/load fencing rather than Save acknowledgement. Full task bodies and current/archive text screened. Current open-PR path census has only personas_screen.py overlap in PR3148@97ac608aba159bd8696ea50d279261867d97e16a, PR3050@2557f186f6e635567c370824fa43e2aadec6b82d and PR3049@bdff2a2d957c9b3ee4cf532589562b901c193df3; no Character editor widget overlaps. Exact upload methods are AST-identical; their older Save methods differ only by absence of the current raw-form tuple change. None preserves later attachments. Bounded recent git log --all symbol/path screening finds only current PR3205 changes. Recheck live heads before filing.

Related tasks: [TASK-100](task-100%20-%20Wire-avatar-upload-in-the-ds-native-character-editor.md); [TASK-481](task-481%20-%20P3b-follow-up-sweep-deferred-editor-polish-minors.md); [TASK-31241](task-31241%20-%20Align-character-conversation-navigation-decisions.md); [TASK-31243](task-31243%20-%20Add-trusted-character-navigation-recovery-and-Roleplay-browse.md); [TASK-33622.14](task-33622.14%20-%20Roleplay-CtrlQ-quits-past-unsaved-Roleplay-drafts-without-the-Save-Discard-Stay-guard.md); [TASK-33910.8](task-33910.8%20-%20Roleplay-frame-B5b-1-one-guard-dialog-per-domain-executors-and-the-load-fence.md); [TASK-36097](task-36097%20-%20Keep-newer-Character-and-Persona-edits-dirty-when-earlier-saves-finish.md).

ADR required: no new ADR for this bounded repair.
ADR paths: [046-roleplay-chat-display-identity-and-template-provenance](../decisions/046-roleplay-chat-display-identity-and-template-provenance.md); [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [150-design-token-system-and-design-language](../decisions/150-design-token-system-and-design-language.md); [161-component-pattern-library](../decisions/161-component-pattern-library.md).
Reason: No new ADR is required for a bounded correction within the existing Character attachment owner. Accepted ADR-046 aggregate-draft amendments explicitly include staged attachments; Accepted application-session-state ADR-033 preserves ownership. Accepted ADR-150/161 govern incidental UI composition only. No new schema, service/provider boundary, global store or mutation-lifetime framework is proposed.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-22-textual-book-after-pr3205.md). The failed assertions establish an unfixed finding, not a passing implementation. Future AC variants remain implementation requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 If a valid avatar upload is staged after an ordinary Character Save captures its submission, completing that Save retains the exact later bytes as an unsaved attachment while adopting the actual persisted record identity and version.
- [ ] #2 The current editor preview/readout and existing aggregate leave/quit draft predicate continue to represent the retained attachment; choosing Stay preserves it without silently acknowledging or replacing it.
- [ ] #3 A subsequent explicit Save persists the retained avatar to the same Character with the correct version and then becomes clean; uploads made before Save persist normally and failed saves retain staged bytes.
- [ ] #4 Mounted real-picker-result and real-store regressions distinguish upload-before-Save, upload-during-pending-Save and same-record retry/failure outcomes while preserving path validation, storage admission, stale-editor fences and existing raw-text/greetings coverage.
<!-- AC:END -->
