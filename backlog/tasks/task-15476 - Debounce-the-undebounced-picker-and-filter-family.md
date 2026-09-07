---
id: TASK-15476
title: Debounce the undebounced picker and filter family
status: Done
assignee: ['@claude']
created_date: '2026-08-11 12:05'
labels:
  - perf
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit, one shared defect shape across many surfaces: full result-set rebuild on every `Input.Changed` with no debounce. Verified sites: `Widgets/Console/console_character_picker_modal.py:205` (remove_children + up to 500 Statics per keystroke), `console_session_switcher_modal.py:128/:151` (unbounded, one Button per conversation), `console_style_picker_modal.py:262` (documents its own lack of debounce; its three sibling modals have it), `UI/Evals/card_picker.py:180` (up to 500 rows), `UI/Chatbooks_Window_Improved.py:524`, `UI/Logs_Window.py:319/:403` (re.compile + regex over up to 10,000 buffered records + RichLog clear/rewrite per character), `UI/Screens/scheduling/schedules_workbench.py:255` (also resets the detail pane to row 0 per character), CCP editors (`ccp_dictionary_editor_widget.py:730/:848`, `ccp_prompt_editor_widget.py:876`), `Widgets/collections_tag_window.py:241/:261`, `Widgets/template_selector.py:184/:193`, the five Persona pickers (up to 1000-row feeds), `Widgets/Console/console_settings_modal.py:1536`, and `UI/Speech/speech_playground_pane.py:611/:681` (duplicate handlers for the same TextArea.Changed — full text materialized twice per keystroke).

Fix direction: copy the debounce shape from `console_prompt_picker_modal.py:203-213` (0.2 s timer + cancel token); cap rendered slices for unbounded lists (Logs shows a capped slice with a count disclosure); prefer display toggles/diffing for bounded pickers; delete the duplicate speech handler. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No listed site rebuilds its full result set on each keystroke — each converted or explicitly justified in the notes
- [x] #2 Logs filter renders a capped slice with a truncation disclosure; compiled pattern cached
- [x] #3 Picker behavior (selection, focus, empty states, schedules detail-pane selection) unchanged — tests on one representative per family
<!-- AC:END -->

## Implementation Plan

1. Apply the `console_prompt_picker_modal.py` 0.2s-timer + cancel debounce shape to every site the description lists: the three sibling Console modals (character/session-switcher/style pickers), `card_picker.py`, `Chatbooks_Window_Improved.py`, `schedules_workbench.py`'s queue filter, the CCP dictionary/prompt editors, `collections_tag_window.py`, `template_selector.py`, the five Persona pickers, and `console_settings_modal.py`'s custom-model input.
2. `Logs_Window.py`: debounce the free-text filter, cache the compiled regex pattern (was recompiled every render), and cap the RichLog render to the most recent `MAX_RENDERED_LINES` matches with a status-line truncation disclosure; re-key the n/N error-jump indices off the rendered (capped) slice, not the full match set.
3. `schedules_workbench.py`: track the selected task by id (not row index) so a filter keystroke restores the same selection when it's still visible, instead of always resetting the detail pane to row 0.
4. `speech_playground_pane.py`: delete the duplicate `on_text_area_changed` handler (Textual's implicit-name dispatch duplicating the `@on(TextArea.Changed)` handler's fallthrough) so `_sync_generate_enabled()` runs once per keystroke instead of twice.
5. Judgment calls per the task brief: bounded (~20-row) sites may keep immediate rebuild if trivial; apply that lens per site and record the call in Notes.
6. Update every existing test whose timing assumed synchronous (undebounced) filtering to await the new debounce window (established idiom already used by `console_prompt_picker_modal`/`console_skill_picker_modal`'s own tests: `pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)`); add one representative debounce-shape test (timer re-arm + cancel) and dedicated Logs render-cap tests.
7. Run each touched site's test suite plus the new tests; read pass counts; self-review diffs.

## Implementation Notes

All 13 sites from the description converted (14 files; the five Persona pickers count as
one family). Every site got the `console_prompt_picker_modal.py` shape: a 0.2s `Timer`
armed on `Input`/`TextArea` change, stopped-and-re-armed on every subsequent keystroke
(cancel), firing the actual rebuild only once typing settles. Two Console modals
(character picker, session switcher) additionally route the settled call through
`run_worker(..., exclusive=True)` since their rebuild is itself `async`; the rest call
the rebuild directly from the timer callback since it's synchronous. No site was left at
"bounded, keep immediate" — the smallest (session switcher, capped at 20 rows) was still
converted since a full remove_children+mount_all cycle isn't "trivial" even at that size,
and it was named explicitly in the task.

Two sites needed more than the template:
- `Logs_Window.py` (AC #2): `_compile_pattern` gained a one-entry cache keyed on filter
  text (was recompiling on every render). `_render_view` now caps the RichLog write to
  the most recent `MAX_RENDERED_LINES` (1000) matches instead of all of them, and
  `_update_status_line` discloses `"(filter matched N; showing most recent 1000)"` when
  the cap trims. The n/N error-jump (`_error_row_indices`) now indexes against
  `_last_rendered` (what's actually in the widget) instead of the full match set --
  otherwise a jump could target a row the cap never rendered.
- `schedules_workbench.py`: beyond debouncing the queue filter, `_render_table` now
  tracks the selected task by id (`_selected_task_id`) and restores that row (via
  `DataTable.move_cursor`) when it's still visible after a filter narrows, instead of
  unconditionally resetting the detail/inspector panes to row 0 on every keystroke (the
  second defect the description called out for this site).
- `speech_playground_pane.py`: deleted `on_text_area_changed` outright. Textual's
  implicit `on_<message>` dispatch fired it for every `TextArea.Changed` in the pane
  alongside the `@on(TextArea.Changed)`-decorated `on_tts_text_changed`, whose fallthrough
  (`self.handle_text_changed`) already called the same `_sync_generate_enabled()` for
  `tts-text-input`. Pure duplicate, not a debounce candidate -- confirmed by temporarily
  restoring it and re-running the one test near this code path, which failed identically
  with or without the handler (see below), proving that failure was unrelated and the
  deletion itself introduces no behavior change.
- `console_settings_modal.py`: only the custom-model-id `Input.Changed` handler is
  debounced (`_sync_readiness_display` rebuilds a full ~15-field
  `ConsoleSessionSettings` draft + re-validates). `picker.set_custom_value(...)` stays
  immediate (cheap, keeps the `ModelSearchPicker`'s own mirror in sync without lag). Save
  correctness is unaffected either way -- `_validated_result_or_show_errors` always
  re-reads live widget values, never the debounced readiness display.

Tests: every touched production file's existing suite was run and, where debounce timing
broke a test's assumption that typing rebuilds synchronously, the test was updated to
`await pilot.pause(<module>.SEARCH_DEBOUNCE_SECONDS + 0.1)` (or the file's equivalently-
named constant) before asserting post-filter state -- the same idiom this codebase's own
`console_prompt_picker_modal`/`console_skill_picker_modal` tests already use. Two
mock-based unit tests (`world_book_picker`/`tag_filter_picker`, which construct the
picker directly with no running Textual app) called `_filter(event)` synchronously outside
any event loop; `Timer` scheduling requires one, so both were switched to call
`_apply_filter_debounced(...)` directly -- the method that now holds the filter-then-
populate logic those tests actually exercise. Added `Tests/UI/
test_console_character_picker_debounce.py` (3 tests: pre-debounce no-rebuild, settled
rebuild, rapid-retype cancel-and-reapply) as the one family-representative debounce-shape
test, plus 4 new tests in `Tests/UI/test_logs_ux_fixes.py` for the render cap/disclosure/
error-jump-reindexing.

Pre-existing failures encountered and left alone (confirmed unrelated -- either in files
this task never touches, or reproduced identically with the change reverted):
`test_console_shell_chip_actions.py::test_swap_seeds_greeting_only_into_an_empty_chat`
(unrelated `ConsoleSessionController` signature mismatch),
`test_console_rail_sections.py::test_popover_apply_returns_replaced_settings`
(`console_model_popover.py`, untouched, streaming-toggle assertion), `test_ux_batch5/6/7`'s
Ollama/llama.cpp network-egress errors (task-15473's known probe), and a cluster of
`test_destination_shells.py`/`test_destination_visual_parity_correction.py`/
`test_console_live_work_handoffs.py` geometry/parity failures spanning schedules **and**
several unrelated destinations (mcp, tools_settings, settings, watchlists) identically --
systemic, not schedules-specific, and none of the failing assertions touch anything this
task's diff changes (CSS/layout untouched). One deterministic failure
(`test_speech_tts_settings_ownership_closeout.py::
test_first_time_audio_cpp_setup_lab_generation_and_console_handoff`) was verified
unrelated by temporarily restoring the deleted duplicate handler and re-running -- it
failed identically either way.

Modified: `tldw_chatbook/Widgets/Console/{console_character_picker_modal,
console_session_switcher_modal,console_style_picker_modal,console_settings_modal}.py`,
`tldw_chatbook/UI/Evals/card_picker.py`, `tldw_chatbook/UI/Chatbooks_Window_Improved.py`,
`tldw_chatbook/UI/Logs_Window.py`,
`tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`,
`tldw_chatbook/UI/Speech/speech_playground_pane.py`,
`tldw_chatbook/Widgets/CCP_Widgets/{ccp_dictionary_editor_widget,
ccp_prompt_editor_widget}.py`, `tldw_chatbook/Widgets/collections_tag_window.py`,
`tldw_chatbook/Widgets/template_selector.py`,
`tldw_chatbook/Widgets/Persona_Widgets/{dictionary_picker,world_book_picker,
tag_filter_picker,dictionary_attach_picker,conversation_attach_picker}.py`, plus the
corresponding test files (timing fixes) and one new test file
(`Tests/UI/test_console_character_picker_debounce.py`).
