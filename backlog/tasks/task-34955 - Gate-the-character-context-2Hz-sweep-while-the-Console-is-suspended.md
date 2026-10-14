---
id: TASK-34955
title: Gate-the-character-context-2Hz-sweep-while-the-Console-is-suspended
status: Done
assignee: []
created_date: 2026-10-13 14:46
labels:
- performance
- ui-perf
- wave-1-timers
dependencies: []
priority: medium
updated_date: 2026-10-13 16:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleCharacterContext arms set_interval(0.5, _sync_progress_counts) in on_mount with no screen-active gate (console_character_context.py:131). The chat screen is reusable (suspended, not unmounted) on navigation away, so this sweep iterates every CharacterConversationButton (fingerprint and dict reads each) at 2Hz while the user is on a completely different screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Interval early-returns or pauses while the owning screen is not current
- [x] Sweep resumes within one tick of returning to the Console (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read console_character_context on_mount + sweep and the chat screen suspend model
2. Early-return the 2Hz sweep while the owning screen is suspended
3. Unit-test the gate
4. Targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_sync_progress_counts` (console_character_context.py) now early-returns when the widget is detached or its owning screen is not the app's current screen. The Console screen is reusable (suspended, not unmounted, on navigation away), so the 2Hz sweep previously walked every CharacterConversationButton while the user was on any other screen; now it no-ops and resumes within one 0.5s tick of returning. Currency gate matches the Console's own pollers (chat_screen) and pauses under modals the same way they do.
- Tests: new self-contained `Tests/UI/test_console_character_context_sweep_gate.py` (3 tests: runs while current, skips while suspended, skips when detached) — 3/3 pass.
- NOTE: `Tests/UI/test_console_character_context.py` errors for ALL 61 tests at this branch's HEAD with the change stashed (same pre-existing mount breakage documented in task-34914's notes); it could not serve as the regression suite.
- ADR required: no — one early-return gate on an existing timer callback.
