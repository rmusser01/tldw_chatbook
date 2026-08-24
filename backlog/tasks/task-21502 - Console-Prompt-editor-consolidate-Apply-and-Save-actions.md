---
id: TASK-21502
title: 'Console Prompt editor: consolidate Apply and Save actions'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 06:50'
labels:
  - console
  - prompts
  - ux
  - responsive
dependencies: []
references:
  - >-
    .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce the structured Prompt editor's action density so applying the working copy is the clear primary outcome and persistence choices remain available through one contextual Save menu. Preserve all existing validation, compatibility, conflict, and lane-application safeguards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The structured editor presents one primary `Apply` action and one `Save…` menu instead of separate Save Prompt, Save Recipe, and Update Original footer buttons.
- [x] #2 The Save menu offers only valid actions for the current artifact: Save as Prompt, Save as Recipe, and Update original when the source is editable and supports an in-place update; unavailable choices are omitted or expose a specific reason.
- [x] #3 System/User replacement choices are presented in a compact pre-Apply summary or step, retain the existing User-on/System-off defaults, and cannot be confused with analysis-context inclusion.
- [x] #4 Apply, save, update, dirty-work, version-conflict, compatibility, and reserved Additional-context behavior remain lossless and fail closed exactly as before.
- [x] #5 At 140x40, 100x30, and 80x24, the final editable block, validation status, Apply action, and Save menu remain scroll-reachable without overlap, clipping, or nested-focus traps.
- [x] #6 Keyboard order reaches the primary action before secondary persistence choices, Escape and Back retain their existing safe-cancel behavior, and every menu action is operable without a pointing device.
- [x] #7 Rendered-frame regression evidence verifies action visibility and painted labels under the production Console hierarchy and stylesheet.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace the three persistence buttons with one keyboard-native Save… selector while preserving typed save/update messages.
2. Keep Apply before Save in focus order and retain the existing lane defaults and validation reasons.
3. Route Console source capabilities into the editor so the Save menu contains only valid Prompt, Recipe, and guarded Update actions with specific unavailable copy.
4. Update focused editor, Console host-gate, conflict, reserved-context, and responsive tests before implementation cleanup.
5. Run targeted Prompt editor/workbench/native verification and regenerate/inspect production-stylesheet captures at 140x40, 100x30, and 80x24.

ADR required: no
ADR path: N/A; ADR-040 remains applicable.
Reason: this consolidates existing commands without changing artifact storage, conditional-update, or Apply transaction boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the three persistence buttons with one native Save selector that emits the existing typed Prompt, Recipe, and guarded-update messages while omitting unsupported choices with specific recovery copy.
- Kept User-on/System-off Apply defaults, made Apply precede Save in focus order, exposed `Ctrl+S`, and left modal Back/Close navigation with the host so the Console editor footer contains exactly Apply plus Save.
- Routed source capability gates into the shared editor and retained validation, compatibility, conflict, mapped Additional-context, and lossless persistence behavior.
- The rendered UAT pass found and fixed an editor-height overlap that visually covered the new action row; production-stylesheet captures now prove separation from the outer footer at 140x40, 100x30, and 80x24.
- Verification: 26 shared-editor tests, 111 Console Prompt modal tests, 4 focused Library persistence tests, the responsive real-app capture stage, Ruff, compilation, and rendered visual inspection passed. No new ADR was required; ADR-040 remains the governing transaction boundary.
<!-- SECTION:NOTES:END -->
