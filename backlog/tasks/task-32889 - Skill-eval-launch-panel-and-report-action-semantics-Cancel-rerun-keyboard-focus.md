---
id: TASK-32889
title: >-
  Skill eval: launch-panel and report action semantics (Cancel, rerun, keyboard
  focus)
status: Done
assignee:
  - '@robert'
created_date: '2026-09-21 22:31'
updated_date: '2026-09-21 23:45'
labels:
  - ux
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review B3+B4+B8 (MEDIUM). (1) Panel 'Cancel' cancels an in-flight run and silently no-ops when idle, but its form placement reads 'close this form'; Escape is unbound so no keyboard close exists. (2) After a successful run the panel is replaced by the report with no 'Run again'; changing depth/models requires re-selecting the bench in the rail. (3) Keyboard-only operation is expert-hostile: Enter on an already-set Select re-opens it instead of advancing; Tab counts drift once dropdowns open/close; the NULL sentinel row sits in keyboard order.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancel relabelled (e.g. 'Stop run') and enabled only while a run is in flight - idle press is no longer a silent no-op with misleading affordance
- [x] #2 A keyboard way exists to close/leave the launch panel (Escape or explicit close) that does not pop the Lab screen
- [x] #3 Report view offers 'Run again' reusing the persisted config; depth/models visible without rail re-selection
- [x] #4 Enter on a Select that already holds a value is a no-op or advances focus (never silently re-opens)
<!-- AC:END -->

## Implementation Notes

- "Cancel" → "Stop run" (same `#skill-eval-cancel` id), mounting disabled; the screen's `_set/_reset_skill_eval_running_ui` arm/disarm it for exactly the in-flight window — idle presses are impossible instead of silent no-ops.
- Escape on the panel posts `CloseRequested`; the screen clears the selection (`select(kind="none")`). Widget-level binding — the Lab-wide "Escape deliberately unbound" screen contract is untouched (its pinning tests stay green).
- `SkillEvalDetail` gains "Run again" (`#skill-eval-run-again`, only when the owning bench resolves from the run rows' task_id) posting `RunAgainRequested(bench_id)`; the screen handler mirrors the RunRequested handler's guards and flag-before-dispatch rule, snapshotting depth/targets from the persisted bench config.
- Panel selects are a `_SetSelect(Select)` subclass: Enter opens the overlay only while no value is held; with a value it's a deliberate no-op (Textual's combined enter/down/space/up binding is replaced wholesale — key subtraction isn't supported). Space/arrows unchanged.
- Tests: `test_stop_run_button_posts_cancel_requested_only_when_enabled`, `test_escape_on_panel_posts_close_requested`, `test_enter_on_a_set_select_no_ops_instead_of_reopening` (panel); `test_escape_on_launch_panel_clears_the_selection`, `test_run_again_on_report_dispatches_with_persisted_config` (screen). Panel 24/24; screen 23/24 (known pre-existing dev red); empty-states + mode-keys green.
- Files: `tldw_chatbook/UI/Evals/skill_eval_panel.py`, `tldw_chatbook/UI/Evals/skill_eval_detail.py`, `tldw_chatbook/UI/Screens/evals_screen.py`, both test files.
