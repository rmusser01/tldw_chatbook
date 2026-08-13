---
id: TASK-15704
title: 'Console: add collapsible status row and left-align restore controls'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-13 06:05'
updated_date: '2026-08-13 15:18'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce Console visual noise by allowing the status-chip row above the composer to collapse while keeping restore controls immediately discoverable at the far left of both collapsed rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can collapse the status-chip row to a one-line `Status hidden` presentation.
- [x] #2 User can restore the status-chip row from a far-left `Status` control, and existing chip state is preserved.
- [x] #3 The collapsed composer restore control appears at the far left while status copy and conditional `Stop` remain usable.
- [x] #4 Status-row collapse state is screen-local and resets when the Console screen is recreated.
- [x] #5 Keyboard focus order and narrow/wide geometry remain correct.
- [x] #6 Relevant automated tests and live Textual verification pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing widget tests for mounted expanded/collapsed status-row presentations and preserved chip state.
2. Implement the minimal `ConsoleStatusChips.set_collapsed()` display toggle without recomposition.
3. Add failing screen tests, then wire screen-local state, button handlers, and inverse-control focus restoration in `ChatScreen`.
4. Update composer geometry expectations first, then move the existing `Expand ▴` child to the far left of the collapsed row.
5. Add and bundle minimal TCSS geometry rules; run focused regressions, layout detection, and isolated live Textual verification.
6. Complete acceptance criteria, implementation notes, and task closeout only after verification passes.

Detailed plan: `Docs/superpowers/plans/2026-08-13-console-row-collapse-controls.md`

Design: `Docs/superpowers/specs/2026-08-12-console-row-collapse-controls-design.md`

ADR required: no

ADR path: N/A

Reason: This is a screen-local UI behavior change following the existing composer ownership pattern; it changes no durable architecture boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added always-mounted expanded/collapsed presentations to `ConsoleStatusChips`, with `ChatScreen` owning the screen-local collapse state and focus handoff. The expanded row keeps the existing chips in an inner horizontal scroller so collapsing does not recompose or discard chip state. The collapsed composer now orders `Expand ▴`, status copy, and conditional `Stop` from left to right. Source TCSS pins both presentations to one row and the generated stylesheet was rebuilt.

Modified production/style files: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/Widgets/Console/console_status_chips.py`, `tldw_chatbook/Widgets/Console/console_composer_bar.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss`, and `tldw_chatbook/css/tldw_cli_modular.tcss`. Focused coverage was added or updated in `Tests/UI/test_console_status_row_collapse.py`, `test_console_composer_collapse.py`, `test_console_chip_strip_overflow.py`, and `test_console_system_prompt_chip.py`.

Verification: the requested eight-file regression command completed with 171 passed, 5 failed, and 2 warnings. Its exact failure set matches the supplied BASE/HEAD baseline: the four `test_console_shell_regions.py` 120x30 geometry cases for `#console-left-rail`, `#console-left-rail-body`, `#console-main-column`, and `#console-run-inspector`, plus `test_console_tab_scope.py::test_console_focus_tour_reaches_transcript_chips_inspector_under_ten_stops` focusing `#console-empty-provider-action`. The known low-rate composer `max_scroll_y == 0` intermittent did not occur. No task-related failure appeared, and no unrelated baseline was changed.

Static/layout evidence: the three changed production Python files compile; Ruff passes all changed Python/tests outside `chat_screen.py`; `chat_screen.py` reports the same 28 pre-existing F401/E713/E731 findings at base `7d14a22cc` and HEAD; Impeccable's layout detector returns `[]` for the three production Python targets plus source TCSS; and `git diff --check` passes for `7d14a22cc..HEAD` and the worktree.

Live Textual verification used an isolated `mktemp -d` profile with `TLDW_TEST_MODE=1`, `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, and `TLDW_CONFIG_PATH` set under `/private/tmp/task15704-live.fySrB4` before imports. Real `ChatScreen` instances mounted against the shipped stylesheet at 100x32 and 140x42. Compositor evidence painted `Status ▴ Status hidden` at rows 28/38 above composer rows 30/40; status restore occupied x=1..10, composer restore x=0..12, and active `Stop` occupied the rightmost eight cells (x=92..100 / 132..140). Enter-key activation returned focus to the inverse status controls and the composer; a safely mocked active-run Stop routed exactly once without expanding; and a recreated, mounted `ChatScreen` reset both screen and widget status-collapse state to expanded.

Self-review of `git diff 7d14a22cc..HEAD` found only the nine planned implementation/test/style files and no additional correctness, security, or scope findings. No new incident-backed reusable lesson surfaced. ADR required: no. ADR path: N/A. Reason: the change follows the existing screen-local composer ownership pattern and introduces no storage, runtime, security, service, or cross-module architecture boundary.
<!-- SECTION:NOTES:END -->
