---
id: TASK-35903
title: >-
  Console system prompt: two tests are red on dev (rail-line click opens
  nothing; CSS pin misses #console-system-prompt-text)
status: To Do
assignee: []
created_date: '2026-10-21 01:27'
labels:
  - console
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two tests in Tests/UI/test_console_system_prompt.py fail on dev 2d071d6d71 and identically on the TASK-34720 branch, under the plain command. test_console_rail_system_line_click_opens_editor_modal: a click on #console-rail-system-line pushes no screen (the stack stays at the default screen plus ChatScreen), so the rail line's route into the system prompt editor has no passing test; the opener itself works when its worker is run directly (TASK-34720 review). test_system_prompt_modal_and_rail_line_css_pinned_in_source_and_bundle: 'missing CSS for #console-system-prompt-text {' in the source stylesheet it reads. Either the product or the tests drifted; find which and fix that side. A sibling pin fails the same way on both sides: Tests/UI/test_console_scope_picker_modal.py::test_scope_picker_css_pinned_in_source_and_bundle ('missing CSS for #console-scope-picker-list {' in the same _agentic_terminal.tcss source), so the CSS pins may share one cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A click on the rail's system-prompt line opens the system prompt editor in the real app, or the test is retargeted to the control that replaced the line, with the reason recorded
- [ ] #2 The CSS pin asserts selectors that exist in both the source stylesheet and the generated bundle, or the missing rule is restored through build_css.py
- [ ] #3 Both tests pass under the plain command
- [ ] #4 test_scope_picker_css_pinned_in_source_and_bundle passes under the plain command, with the same fix or its own recorded reason
<!-- AC:END -->
