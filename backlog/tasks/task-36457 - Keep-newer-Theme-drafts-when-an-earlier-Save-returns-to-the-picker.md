---
id: TASK-36457
title: Keep newer Theme drafts when an earlier Save returns to the picker
status: In Progress
created_date: 2026-10-23 16:39
labels:
- textual
- book-audit
priority: medium
references:
- backlog/tasks/task-32948 - Settings-Theme-picker-first-redesign.md
- backlog/tasks/task-32941 - Leaving-Theme-with-unsaved-edits-asks-Save-Discard-Stay.md
- backlog/tasks/task-32949 - Leaving-Settings-with-unsaved-theme-edits-drops-them-silently.md
- backlog/tasks/task-33078 - Theme-editor-file-actions-scan-off-the-UI-thread.md
- backlog/tasks/task-33240 - Theme-editor-Edit-and-Reset-read-theme-files-off-the-UI-thread.md
- backlog/tasks/task-31250 - Theme-editor---saved-user-themes-are-never-loaded-so-Save-cannot-make-a-theme-the-launch-theme.md
- backlog/tasks/task-31251 - Theme-editor---Name-box-is-write-only-so-Apply-Export-Reset-and-Delete-act-on-a-stale-name.md
- backlog/tasks/task-31258 - Theme-editor---Save-and-Export-overwrite-existing-files-silently.md
- tldw_chatbook/Widgets/settings_theme_editor.py
- tldw_chatbook/Widgets/settings_theme_picker.py
documentation:
- backlog/decisions/011-chatbook-workbench-ui-system.md
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/033-settings-commit-models-three-honestly-labeled.md
- backlog/decisions/150-design-token-system-and-design-language.md
- backlog/decisions/161-component-pattern-library.md
- Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md
assignee:
- '@codex'
updated_date: 2026-10-23 20:37
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Theme Save correctly leaves a later palette dirty, but its Saved message still hides the editor. Ordinary Edit then reloads the earlier saved TOML and clears the hidden draft without a prompt. Keep later accepted theme edits recoverable and preserve ordinary Save return behavior when no later draft exists.

Priority: Medium (P2), rank 4 of five in this audit. See the report for impact and ordering.

Observed evidence: One real TOML write stores submitted #112233; actual later Pilot input/model is #AABBCC and dirty. Completion shows picker with hidden dirty #AABBCC; ordinary Edit reopens #112233 clean. Matched ordinary control passes. Production Save-button handler is invoked directly; later input is actual Pilot. Name/dark/Save-as/next Save are future regression requirements.

Existing scope: TASK-32948 owns ordinary Save-to-picker and explicit Back. TASK-32941/TASK-32949 protect explicit leave actions; TASK-33078 moves writes off-loop; TASK-33240 fences reads/Reset/new sessions. None owns same-session Save auto-return hiding a later accepted draft. Preserve those contracts and existing file-target rules.

Related tasks: [TASK-32948](task-32948%20-%20Settings-Theme-picker-first-redesign.md), [TASK-32941](task-32941%20-%20Leaving-Theme-with-unsaved-edits-asks-Save-Discard-Stay.md), [TASK-32949](task-32949%20-%20Leaving-Settings-with-unsaved-theme-edits-drops-them-silently.md), [TASK-33078](task-33078%20-%20Theme-editor-file-actions-scan-off-the-UI-thread.md), [TASK-33240](task-33240%20-%20Theme-editor-Edit-and-Reset-read-theme-files-off-the-UI-thread.md), [TASK-31250](task-31250%20-%20Theme-editor---saved-user-themes-are-never-loaded-so-Save-cannot-make-a-theme-the-launch-theme.md), [TASK-31251](task-31251%20-%20Theme-editor---Name-box-is-write-only-so-Apply-Export-Reset-and-Delete-act-on-a-stale-name.md), [TASK-31258](task-31258%20-%20Theme-editor---Save-and-Export-overwrite-existing-files-silently.md).

ADR check: no new ADR expected for a bounded repair within existing owners and contracts. Applicable decisions: [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md), [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md), [033-settings-commit-models-three-honestly-labeled](../decisions/033-settings-commit-models-three-honestly-labeled.md), [150-design-token-system-and-design-language](../decisions/150-design-token-system-and-design-language.md), [161-component-pattern-library](../decisions/161-component-pattern-library.md). Re-evaluate if implementation changes a boundary.

[Textual book audit: evidence, controls, limits and overlap review](../../Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md). The audit records the pre-fix behavior; implementation evidence is recorded below.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Palette, name and dark-mode edits accepted while a Save is pending remain recoverable and honestly dirty after completion; only the submitted theme is written.
- [x] #2 The actual saved-file identity remains separate from a newer draft name. A subsequent explicit Save follows the existing target-resolution and overwrite rules and persists the retained draft.
- [x] #3 A successful Save with no later edits still returns to the picker. Existing explicit Back, category/destination leave, backup and quit protections remain effective.
- [x] #4 Mounted real-TOML controls cover ordinary Save, later palette/name/dark edits, definite failure, Save-as and next Save, witnessing actual enabled-control input before release and ordinary reopening behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the Theme editor Save completion, picker return, file identity and existing leave guards; reproduce pending-save draft loss in mounted real-TOML tests with enabled-control input witnessed before release.
2. Preserve newer palette/name/dark raw input and dirty state in the existing editor owner, retain actual saved-file identity, and return to the picker only for the submitted unchanged draft; preserve Save-as and next-Save overwrite rules.
3. Run the focused mounted regressions and existing Theme/leave/backup/quit controls, record RED/GREEN evidence and update the theme-system documentation and implementation notes.
ADR required: no
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: bounded asynchronous completion repair within existing editor/picker ownership and file/leave contracts; no new boundary, storage model, dependency or visual token.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Theme Save keeps later raw palette/name/dark edits visibly dirty and tracks the actual saved file separately from a newer draft name. Save-as changes only an unchanged Name box; the next Save preserves existing file-resolution and overwrite rules. Session/raw-snapshot checks protect the posted Saved handoff before the picker returns, including queued Changed input. The theme-system architecture document records the behavior.

All13 mounted real-TOML regressions pass. Existing controls:17 of18 pass directly and five Reset controls pass. The fixed-pause Save-as highlight test also fails on pristine HEAD before its catalog scan completes; the bounded final-source scan witness preserves every assertion and passes. Existing test source is unchanged. See the [verification report](../../Docs/superpowers/qa/2026-10-23-textual-pending-draft-followups.md) for exact receipts and qualification. Existing ADR-011, both applicable ADR-033 records and ADR-150/161 apply; no new ADR. Independent Superpowers review approved the implementation; required current-head PR #3228 CI is pending, so the task stays In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->