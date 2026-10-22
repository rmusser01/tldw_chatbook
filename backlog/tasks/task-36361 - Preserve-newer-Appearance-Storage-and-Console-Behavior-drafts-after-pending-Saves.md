---
id: TASK-36361
title: Preserve newer Appearance Storage and Console Behavior drafts after pending
  Saves
status: Done
created_date: 2026-10-22 20:43
labels:
- settings
- textual-patterns
- reliability
priority: medium
references:
- backlog/decisions/033-settings-commit-models-three-honestly-labeled.md
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/004-settings-storage-defaults-restart-boundary.md
- backlog/decisions/161-component-pattern-library.md
documentation:
- Docs/superpowers/qa/2026-10-22-textual-retry-and-pending-drafts.md
assignee:
- '@codex'
updated_date: 2026-10-22 23:51
modified_files:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Settings_Modules/staged_save_ack.py
- Tests/UI/test_settings_pending_save_drafts.py
- Tests/UI/test_settings_configuration_hub.py
- Tests/UI/test_settings_context_memory_controls.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A successful staged Settings Save must acknowledge only the submitted values while later accepted edits remain visible and unsaved in Appearance, Storage and Console Behavior.

### Evidence and scope

Medium priority, audit order 2 of 4. Mounted actual Save workers held the real private TOML writer. Appearance submittedfont18/later19 persists18 then resets the same Input to18/clean; Storage submitted.db/later.db persists submitted.db then resets clean; Console paste threshold120/later121 persists120 then resets clean. Three unchanged controls pass and three later-edit retention assertions fail. Input identity/focus remain; this is draft loss, not a recompose/focus finding. Initial Console enum-only setup failures were corrected and excluded. Natural disk incidence, next Save and failure/revert controls were not measured.

Source: UI/Screens/settings_screen.py completion methods at34640,35052,35082; unconditional draft pops at34656,35059,35111, with existing workers at34671,35072,35141 (under tldw_chatbook/).

Book basis: Chapter7 printed144-145/PDF153-154 covers programmatic change messages; printed149/PDF158 and chapter17 printed326/PDF335 guide completion applicability. Chapter9 printed170-171/PDF179-180 and180/PDF189 informs editable-child checks; these witnesses retain the same child controls.

### Existing ownership and decisions

Completed36098 is explicitly Image/Video only;32190 Advanced Config. Ordinary staged workflow qualification32757/32761 and origin/queued-input35680/35867 do not own removal of already-staged newer values by these three finishers. Open2831 is focus intent,35013 save-model labeling,35022 Ctrl+S routing and32804.12 blocking work. All three affected completions/workers are AST-identical on four current overlapping PR heads.

Related tasks: [TASK-36098](task-36098%20-%20Preserve-newer-Image-and-Video-Settings-drafts-after-pending-saves.md); [TASK-32190](task-32190%20-%20Preserve-Advanced-Config-drafts-and-guard-draft-replacement.md); [TASK-32757](task-32757%20-%20Qualify-Appearance-Theme-and-Splash-Settings-workflows.md); [TASK-32761](task-32761%20-%20Qualify-Console-Behavior-and-Storage-Settings-workflows.md); [TASK-80](task-80%20-%20Functionalize-Settings-Appearance-defaults.md); [TASK-81](task-81%20-%20Functionalize-Settings-Storage-defaults.md); [TASK-35676](task-35676%20-%20Keep-Console-Behavior-form-refreshes-from-staging-user-edits.md); [TASK-35677](task-35677%20-%20Give-Settings-composer-paste-controls-local-widget-ownership.md); [TASK-35680](task-35680%20-%20Keep-parallel-run-form-refreshes-out-of-Settings-drafts.md); [TASK-35867](task-35867%20-%20Keep-remaining-Console-input-refreshes-out-of-user-drafts.md); [TASK-2831](task-2831%20-%20Settings-harden-category-focus-intent-against-recompose-races.md); [TASK-35013](task-35013%20-%20Settings-shows-which-controls-save-without-s-and-where-each-page-s-saves-go.md); [TASK-35022](task-35022%20-%20Ctrl-S-saves-a-Settings-page-from-inside-its-fields.md); [TASK-32804.12](task-32804.12%20-%20Close-the-remaining-synchronous-work-on-the-loop-findings.md).

ADR required: no new ADR for the bounded correction.
ADR paths: [033-settings-commit-models-three-honestly-labeled](../decisions/033-settings-commit-models-three-honestly-labeled.md); [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [004-settings-storage-defaults-restart-boundary](../decisions/004-settings-storage-defaults-restart-boundary.md); [161-component-pattern-library](../decisions/161-component-pattern-library.md).
Reason: No new ADR is expected for submitted-only acknowledgement in existing staged category owners. Preserve all three commit models and Storage restart-only defaults; no global draft store, live migration or handle reconnection.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-22-textual-book-after-pr3207.md). These are findings and future regression requirements, not implemented fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In all three categories, successful Save persists its submitted snapshot and advances the saved baseline while later raw edits remain visible and visibly unsaved, including accepted edits whose Changed messages are still queued.
- [x] #2 A subsequent explicit Save commits the retained draft, unchanged successful Save becomes clean normally, genuine failure retains drafts, and confirmed Revert restores the actual saved baseline.
- [x] #3 Save completion preserves current editing context without stealing a later focus move; no unrelated field is reset, including when an edit returns to a value from the prior baseline.
- [x] #4 Bounded mounted real-writer tests cover each sibling, same/different-field later edits and ordinary/failure/revert controls while retaining existing event-origin, validation and Storage restart-only boundaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: existing ADR paths in this task. Reason: bounded correction within the current owner; no new storage, provider or application boundary.

1. Read the linked task, ADRs, current owner tests and retained audit evidence.
2. Add and run mounted behavior regressions RED, then implement the smallest owner-local correction: Reuse the existing media Save acknowledgement pattern where appropriate. Advance the persisted baseline for Appearance, Storage and Console Behavior from the submitted snapshot while retaining newer raw control values, including accepted changes whose Changed message is still queued. Do not refresh away focus/selection or unrelated fields.
3. Verify targeted new/adjacent tests and static checks; obtain independent Superpowers task review.
4. Record evidence, update notes/ACs and relevant documentation, then mark Done via CLI.

Implementation plan: Docs/superpowers/plans/2026-10-22-textual-retry-and-pending-drafts.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Acknowledge submitted Appearance, Storage and Console Behavior snapshots while preserving current live raw controls, queued edits, focus and selection. The lazy same-owner staged_save_ack helper advances saved originals without resynchronizing editors; two existing private-worker spies accept the explicit submitted snapshot. New19mounted real-writer cases pass, final4actual-writer/format checks pass, adjacent29pass plus one exact-base-qualified pre-Settings startup refusal. No new Ruff diagnostics; inherited full-file formatting/lint debt is documented and changed ranges are clean. Lowered Settings size ratchet35668→35623; no raised budgets or handle reconnection. No new ADR: existing ADR-033 commit models/session ownership, ADR-004 restart boundary and ADR-161 apply. Independent Astra Superpowers task review Approved; all19newnodes enrolled. See Docs/superpowers/qa/2026-10-22-textual-retry-and-pending-drafts.md for complete qualifications.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
