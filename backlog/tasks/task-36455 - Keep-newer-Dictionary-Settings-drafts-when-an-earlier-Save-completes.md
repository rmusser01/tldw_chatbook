---
id: TASK-36455
title: Keep newer Dictionary Settings drafts when an earlier Save completes
status: Done
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
assignee:
- '@codex'
updated_date: 2026-10-24 10:27
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A pending Dictionary Settings Save overwrites text accepted afterward and clears both the form and aggregate dirty predicates. Real SQLite stores only the submitted settings, so the later edit is lost immediately. Acknowledge the submitted settings and committed version without replacing newer live controls.

Priority: Medium (P2), rank 2 of five in this audit. See the report for impact and ordering.

Observed evidence: Submitted name Submitted; actual later Input text Submitted-new; real SQLite v2 stores Submitted; completion restores Submitted and clears screen/aggregate dirty. Matched ordinary control persists Submitted-new cleanly. Only update/name measured; other controls, next Save, failure and navigation/quit remain regression requirements.

Existing scope: TASK-33910.14 remains To Do for entry/Options draft domains, transitions and creation; its full ACs do not cover ordinary pending Options Save acknowledgement. TASK-33910.8 owns transition/load fences. Geometry and keybinding tasks remain separate. TASK-36097 protects Character/Persona forms, not Dictionary Settings.

Related tasks: [TASK-33910.14](task-33910.14%20-%20Roleplay-frame-B9a-entry-and-Options-draft-domains-no-layout-change.md), [TASK-33910.8](task-33910.8%20-%20Roleplay-frame-B5b-1-one-guard-dialog-per-domain-executors-and-the-load-fence.md), [TASK-33785](task-33785%20-%20Roleplay-lore-and-dictionary-Settings-tabs-cannot-scroll-so-Save-settings-Export-and-Enabled-are-unreachable.md), [TASK-33786](task-33786%20-%20Roleplay-dictionary-and-lore-entry-forms-push-most-controls-off-screen-and-label-none.md), [TASK-33910.17](task-33910.17%20-%20Roleplay-frame-B11-editors-growth-caps-sections-CtrlS-in-Look-and-Tool-policy.md), [TASK-36097](task-36097%20-%20Keep-newer-Character-and-Persona-edits-dirty-when-earlier-saves-finish.md), [TASK-36131](task-36131%20-%20Preserve-avatars-uploaded-while-an-earlier-Character-Save-is-pending.md), [TASK-481](task-481%20-%20P3b-follow-up-sweep-deferred-editor-polish-minors.md).

ADR check: no new ADR expected for a bounded repair within existing owners and contracts. Applicable decisions: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md), [046-roleplay-chat-display-identity-and-template-provenance](../decisions/046-roleplay-chat-display-identity-and-template-provenance.md), [115-personas-demand-mounted-center-views](../decisions/115-personas-demand-mounted-center-views.md), [079-workspace-assistant-defaults](../decisions/079-workspace-assistant-defaults.md). Re-evaluate if implementation changes a boundary.

[Textual book audit: evidence, controls, limits and overlap review](../../Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md). The audit records the pre-fix behavior; implementation evidence is recorded below.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary Dictionary Settings Save adopts the actual persisted version while later raw settings edits remain visible and unsaved; only submitted settings are acknowledged.
- [x] #2 The next explicit Save persists retained settings to the same dictionary using the correct optimistic-lock version. No-later-edit success becomes clean and definite failures preserve the current draft for retry.
- [x] #3 Existing New, leave and quit protections continue to detect retained settings edits, and choosing Stay retains them; entry-draft and transition ownership remain with their existing tasks.
- [x] #4 Mounted real-service/SQLite regressions cover ordinary success, input before Save, input accepted during Save, next Save and definite failure, witnessing child delivery independently of pump settlement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the ordinary pending Dictionary Settings save race with a mounted Personas screen and real local scope service/SQLite, retaining immutable RED receipts and witnessing raw child input delivery independently of the blocked parent pump.
2. Compare existing form acknowledgement patterns, then acknowledge only submitted settings and the persisted optimistic-lock version without overwriting newer raw settings controls or clearing their dirty predicates.
3. Verify mounted controls for ordinary/no-later-edit success, input before Save, input during Save including queued Changed delivery, a next Save to the same dictionary using the committed version, and definite failure preserving the draft for retry.
4. Exercise existing New, leave and quit guards with retained settings edits and Stay; preserve entry/transition ownership. Run only targeted dictionary and neighboring guard regressions, inspect diffs and record exact RED/GREEN receipts and completion draft notes.
ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md; backlog/decisions/115-personas-demand-mounted-center-views.md; backlog/decisions/079-workspace-assistant-defaults.md
Reason: bounded acknowledgement bug fix within existing dictionary form, screen, persistence and guard owners; no schema, boundary, policy or architecture change.
CI aggregate follow-up: the extracted completion has one new W002 post-await Library-pane lookup. Add a mounted real-Save/SQLite pane-removal RED control, replace the unsafe lookup with safe bounded publication, preserve ordinary and newer-draft behavior, append the control to the existing required census, and run the guard/compatibility/boot checks plus fresh Superpowers review. Do not expand the safety census. ADR required: no; existing ADR-033/011 state/UI ownership and ADR-097 apply. See the latest 2026-10-23 pending-draft plan section.
Live-dev B2a integration: PR #3229 replaces ListView row widgets with stable OptionList item IDs. Preserve its incoming production/list/census/CSS behavior. Reproduce the old Dictionary fixture contract failure, migrate only fixture selection to the existing kind-qualified B2a helpers, retain every acceptance assertion and all13 cases, then run mounted Dictionary and affected list/CI/boot/size/source-guard controls. The combined Personas ceiling is lowered from incoming16885 to measured16878 (same7-line extraction); no budget increase for our change. Fresh scoped Superpowers review and new normal final-head CI qualify integration. ADR required:no; existing ADR033/115/011/103/097 and TASK33910.3 apply.
<!-- SECTION:PLAN:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and independently reviewed in PR #3228; targeted acceptance evidence complete. Final hosted CI is followed in the PR. Existing task/ADR links retained; no new ADR.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Dictionary Settings Save acknowledges the submitted raw controls without reloading over newer input, adopts the actual optimistic-lock version, and recomputes dirty state when older notifications arrive. Name/description, strategy, enabled and unfinished budget edits remain visible for the next Save to the same dictionary. Existing Persona completion is unchanged; its shared completion helper hosts the Dictionary tail. PersonasScreen drops to 16,554 lines. Entry and transition ownership and existing New/leave/quit guards stay in place.

Twelve mounted real-service/SQLite regression cases pass, with next-Save/version, queued notification, failure and Stay controls; eight existing widget and 17 screen contracts pass with the documented bootstrap qualification. See the [verification report](../../Docs/superpowers/qa/2026-10-23-textual-pending-draft-followups.md) for exact evidence and limits. Hosted UI3 exposed a fixture selecting a stale Character row before Dictionary list replacement. It now waits for the exact kind-qualified Dictionary row and selected kind/ID/detail. All 12 case bodies/IDs stay unchanged; four native readiness/Save/guard controls pass (4/4). Character-chat architecture documentation is updated. Existing ADR-033 application state and ADR-046/079/115 apply; no new ADR. Independent Superpowers review approved the implementation.

Implementation acceptance and independent review are complete in [PR #3228](https://github.com/rmusser01/tldw_chatbook/pull/3228). All changed production files are identical to the reviewed rebased head whose hosted [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40020015425) passed; the subsequent Dictionary fixture repair passed four native controls and scoped review. Final task/documentation metadata is bundled before the final CI run. Hosted checks remain a delivery gate and their final exact-head results are recorded in the PR check rollup and description. Targeted local verification only; no full local suite.
Required-aggregate follow-up: the extracted completion now treats a missing Library pane after row rendering as an ended presentation context. Real SQLite Save identity/version and raw-draft acknowledgement remain unchanged. A mounted actual-Save control witnesses committed version2 and pane removal before releasing completion, then proves later typing stays usable; matched RED raises NoMatches and GREEN passes. The obsolete old-wrapper two-site W002 pin is removed; no new unsafe row is allowed. The complete targeted selection passes213 cases, and the exact native worker guard passes. Existing ADR-033/011/097 and the linked QA report cover this bounded lifetime repair; fresh scoped Superpowers review and final-head hosted CI remain the delivery gate.
Scoped fresh-context Superpowers review of e9692090 found no Critical, Important or Minor findings and approved publication for fresh CI. Matched runtime RED/GREEN and the 213-case targeted selection qualify the two aggregate repairs; post-rebase CI/workflow contracts pass 224 cases (37.453s JUnit). Production and scripts are unchanged across rebase onto dev 1381e70735. Eleven exact aggregate commands pass natively; unchanged Canvas reproduction requires hosted Python 3.12.11. Final-head hosted CI and paginated feedback remain the delivery gate in PR #3228. Existing ADRs apply; no new ADR.
Live-dev integration onto ae85e59e8f (PR #3229/TASK-33910.3) preserves the B2a OptionList/window implementation. Only this task fixture migrates from retired row widgets to existing kind-qualified item membership and selection-action helpers; all test bodies/parameters/acceptance assertions remain AST-identical. All13 Dictionary cases pass across native387+11 qualifying cases, including current list/CI/boot/size controls. The sole rebase conflict lowers the combined Personas ceiling from incoming16885 to actual16878; no budget grows. Eleven exact aggregate commands pass; unchanged Canvas requires hostedPython3.12.11. Fresh scoped Superpowers review of8e06b3bb found no Critical/Important/Minor issues and approved publication for freshCI. Existing ADR033/115/011/103/097 and QA report apply; no newADR. Final-head CI/feedback remain the PR delivery gate.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->