---
id: TASK-16503
title: Audit remaining Select.BLANK usages outside settings_screen
status: To Do
assignee: []
created_date: '2026-08-15 16:10'
labels:
  - tech-debt
  - textual-upgrade
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-565 swept `Select.BLANK` out of settings_screen.py after establishing that on Textual 8.x it silently resolves to `Widget.BLANK == False` instead of the real blank sentinel `Select.NULL`. TASK-16502 fixed the crashing site in console_model_popover.py. Roughly 60 usages across ~20 other files remain, in three distinct behavior classes that need different treatment:

1. Sentinel-intent comparisons that are now dead (`value == Select.BLANK` never matches a genuine blank selection): STTS_Window.py, character_voice_widget.py, media_viewer_panel.py, compact_model_bar.py, Outputs_Panel.py, library_note_folder_dialog.py (`value is not Select.BLANK`), speech mixins, Tools_Settings_Window.py (deprecated, nav-unreachable), bench_editor.py. These silently misbehave and should become `Select.NULL`.
2. Assignments `select.value = Select.BLANK` (Utils/ui_helpers.py:75/112, Outputs_Panel.py:208, mcp_inspector.py:2737) — assigning `False` raises `InvalidSelectValueError` at that line unless the options deliberately contain the `False` placeholder; ui_helpers' broad `except Exception` currently downgrades this to an error log.
3. Deliberate `False`-placeholder option values with explanatory comments (mcp_rail.py, mcp_inspector.py, mcp_server_mutations.py, Study_Window.py + flashcards/quizzes handlers) — these treat `Select.BLANK` as a synthetic option value on purpose and MUST NOT be blindly renamed to `Select.NULL`; if touched, introduce a named module-level placeholder constant instead.

A blind find-and-replace is wrong for class 3; each site needs classification first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Every remaining `Select.BLANK` usage is classified as sentinel-intent, crashing assignment, or deliberate placeholder, with the classification recorded in the task
- [ ] #2 Sentinel-intent comparisons and crashing assignments behave correctly against the real `Select.NULL` sentinel, with regression coverage for the user-reachable ones
- [ ] #3 Deliberate placeholder sites either keep their current behavior or move to a named constant; no behavior change ships without a test
<!-- AC:END -->
