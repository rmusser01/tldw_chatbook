---
id: TASK-36455
title: Keep newer Dictionary Settings drafts when an earlier Save completes
status: To Do
created_date: 2026-10-23 16:39
labels:
- textual
- book-audit
priority: medium
references:
- backlog/tasks/task-33910.14 - Roleplay-frame-B9a-entry-and-Options-draft-domains-no-layout-change.md
- backlog/tasks/task-33910.8 - Roleplay-frame-B5b-1-one-guard-dialog-per-domain-executors-and-the-load-fence.md
- backlog/tasks/task-33785 - Roleplay-lore-and-dictionary-Settings-tabs-cannot-scroll-so-Save-settings-Export-and-Enabled-are-unreachable.md
- backlog/tasks/task-33786 - Roleplay-dictionary-and-lore-entry-forms-push-most-controls-off-screen-and-label-none.md
- backlog/tasks/task-33910.17 - Roleplay-frame-B11-editors-growth-caps-sections-CtrlS-in-Look-and-Tool-policy.md
- backlog/tasks/task-36097 - Keep-newer-Character-and-Persona-edits-dirty-when-earlier-saves-finish.md
- backlog/tasks/task-36131 - Preserve-avatars-uploaded-while-an-earlier-Character-Save-is-pending.md
- backlog/tasks/task-481 - P3b-follow-up-sweep-deferred-editor-polish-minors.md
- tldw_chatbook/UI/Screens/personas_screen.py
- tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py
documentation:
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md
- backlog/decisions/115-personas-demand-mounted-center-views.md
- backlog/decisions/079-workspace-assistant-defaults.md
- Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A pending Dictionary Settings Save overwrites text accepted afterward and clears both the form and aggregate dirty predicates. Real SQLite stores only the submitted settings, so the later edit is lost immediately. Acknowledge the submitted settings and committed version without replacing newer live controls.

Priority: Medium (P2), rank 2 of five in this audit. See the report for impact and ordering.

Observed evidence: Submitted name Submitted; actual later Input text Submitted-new; real SQLite v2 stores Submitted; completion restores Submitted and clears screen/aggregate dirty. Matched ordinary control persists Submitted-new cleanly. Only update/name measured; other controls, next Save, failure and navigation/quit remain regression requirements.

Existing scope: TASK-33910.14 remains To Do for entry/Options draft domains, transitions and creation; its full ACs do not cover ordinary pending Options Save acknowledgement. TASK-33910.8 owns transition/load fences. Geometry and keybinding tasks remain separate. TASK-36097 protects Character/Persona forms, not Dictionary Settings.

Related tasks: [TASK-33910.14](task-33910.14%20-%20Roleplay-frame-B9a-entry-and-Options-draft-domains-no-layout-change.md), [TASK-33910.8](task-33910.8%20-%20Roleplay-frame-B5b-1-one-guard-dialog-per-domain-executors-and-the-load-fence.md), [TASK-33785](task-33785%20-%20Roleplay-lore-and-dictionary-Settings-tabs-cannot-scroll-so-Save-settings-Export-and-Enabled-are-unreachable.md), [TASK-33786](task-33786%20-%20Roleplay-dictionary-and-lore-entry-forms-push-most-controls-off-screen-and-label-none.md), [TASK-33910.17](task-33910.17%20-%20Roleplay-frame-B11-editors-growth-caps-sections-CtrlS-in-Look-and-Tool-policy.md), [TASK-36097](task-36097%20-%20Keep-newer-Character-and-Persona-edits-dirty-when-earlier-saves-finish.md), [TASK-36131](task-36131%20-%20Preserve-avatars-uploaded-while-an-earlier-Character-Save-is-pending.md), [TASK-481](task-481%20-%20P3b-follow-up-sweep-deferred-editor-polish-minors.md).

ADR check: no new ADR expected for a bounded repair within existing owners and contracts. Applicable decisions: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md), [046-roleplay-chat-display-identity-and-template-provenance](../decisions/046-roleplay-chat-display-identity-and-template-provenance.md), [115-personas-demand-mounted-center-views](../decisions/115-personas-demand-mounted-center-views.md), [079-workspace-assistant-defaults](../decisions/079-workspace-assistant-defaults.md). Re-evaluate if implementation changes a boundary.

[Textual book audit: evidence, controls, limits and overlap review](../../Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md). This task does not claim the repair is implemented.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ordinary Dictionary Settings Save adopts the actual persisted version while later raw settings edits remain visible and unsaved; only submitted settings are acknowledged.
- [ ] #2 The next explicit Save persists retained settings to the same dictionary using the correct optimistic-lock version. No-later-edit success becomes clean and definite failures preserve the current draft for retry.
- [ ] #3 Existing New, leave and quit protections continue to detect retained settings edits, and choosing Stay retains them; entry-draft and transition ownership remain with their existing tasks.
- [ ] #4 Mounted real-service/SQLite regressions cover ordinary success, input before Save, input accepted during Save, next Save and definite failure, witnessing child delivery independently of pump settlement.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
