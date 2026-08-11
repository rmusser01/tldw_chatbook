---
id: TASK-15476
title: Debounce the undebounced picker and filter family
status: To Do
assignee: []
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
- [ ] #1 No listed site rebuilds its full result set on each keystroke — each converted or explicitly justified in the notes
- [ ] #2 Logs filter renders a capped slice with a truncation disclosure; compiled pattern cached
- [ ] #3 Picker behavior (selection, focus, empty states, schedules detail-pane selection) unchanged — tests on one representative per family
<!-- AC:END -->
