---
id: TASK-21502
title: 'Console Prompt editor: consolidate Apply and Save actions'
status: To Do
assignee: []
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 04:46'
labels:
  - console
  - prompts
  - ux
  - responsive
dependencies: []
references:
  - .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce the structured Prompt editor's action density so applying the working copy is the clear primary outcome and persistence choices remain available through one contextual Save menu. Preserve all existing validation, compatibility, conflict, and lane-application safeguards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The structured editor presents one primary `Apply` action and one `Save…` menu instead of separate Save Prompt, Save Recipe, and Update Original footer buttons.
- [ ] #2 The Save menu offers only valid actions for the current artifact: Save as Prompt, Save as Recipe, and Update original when the source is editable and supports an in-place update; unavailable choices are omitted or expose a specific reason.
- [ ] #3 System/User replacement choices are presented in a compact pre-Apply summary or step, retain the existing User-on/System-off defaults, and cannot be confused with analysis-context inclusion.
- [ ] #4 Apply, save, update, dirty-work, version-conflict, compatibility, and reserved Additional-context behavior remain lossless and fail closed exactly as before.
- [ ] #5 At 140x40, 100x30, and 80x24, the final editable block, validation status, Apply action, and Save menu remain scroll-reachable without overlap, clipping, or nested-focus traps.
- [ ] #6 Keyboard order reaches the primary action before secondary persistence choices, Escape and Back retain their existing safe-cancel behavior, and every menu action is operable without a pointing device.
- [ ] #7 Rendered-frame regression evidence verifies action visibility and painted labels under the production Console hierarchy and stylesheet.
<!-- AC:END -->

