---
id: TASK-1334
title: Toggle selected Console message off on repeated activation
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31 22:16'
updated_date: '2026-07-31 23:18'
labels:
  - console
  - messages
  - ux
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-31-console-message-selection-toggle-design.md
  - Docs/superpowers/plans/2026-07-31-console-message-selection-toggle.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console users clear the active transcript message by activating that same message again with either the mouse or keyboard, so contextual actions can be dismissed without moving to another target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clicking an already selected transcript message clears selection and hides its contextual action row.
- [x] #2 Pressing Enter while a transcript message is selected clears selection and hides its contextual action row.
- [x] #3 Pressing Enter with no selected message still selects the first transcript message.
- [x] #4 Arrow-key navigation and contextual action activation retain their existing behavior.
- [x] #5 Focused automated regressions cover the mouse and keyboard toggle paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Routine interaction correction within the existing Console transcript selection boundary; no storage, ownership, service-contract, security, dependency, or long-lived application-structure decision changes.

Design: Docs/superpowers/specs/2026-07-31-console-message-selection-toggle-design.md
Implementation plan: Docs/superpowers/plans/2026-07-31-console-message-selection-toggle.md

1. Add a failing mounted regression for clicking the selected transcript message again.
2. Add a validated toggle_message_selection() API, route message-row clicks through it, and verify the mouse selection regressions.
3. Replace the ambiguous keyboard test with explicit Enter-select, Enter-deselect, boundary-navigation, and focused-action-button coverage; observe the selected-message Enter test fail.
4. Route transcript Enter through the toggle API while preserving no-selection Enter and idempotent navigation behavior.
5. Run the focused transcript module, broader selected-message Console regressions, and git diff --check.
6. Self-review, complete all acceptance criteria, add implementation notes, and set TASK-1334 to Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `ConsoleTranscript.toggle_message_selection()` as the explicit activation toggle while preserving idempotent `select_message()` for navigation and internal callers. Repeated message-row clicks and Enter on the selected transcript message now clear selection through the existing refresh/notification path; Enter with no selection, boundary navigation, and focused contextual action buttons retain their prior behavior.

Added mounted Textual regressions for mouse, keyboard, boundary-navigation, and focused-action paths. Updated the existing keyboard-copy flow to remove its obsolete preliminary Enter, which now correctly means deselect.

Verification: 57 focused transcript tests passed; 16 broader selected-message/message-action tests passed with 177 deselected. Both runs emitted the two pre-existing Requests/webrtcvad dependency warnings. `git diff --check` passed.

ADR required: no. This routine UI interaction correction stays within the existing Console transcript selection boundary.

Modified: `tldw_chatbook/Widgets/Console/console_transcript.py`, `Tests/UI/test_console_native_transcript.py`, and `Tests/UI/test_console_native_chat_flow.py`.
<!-- SECTION:NOTES:END -->
