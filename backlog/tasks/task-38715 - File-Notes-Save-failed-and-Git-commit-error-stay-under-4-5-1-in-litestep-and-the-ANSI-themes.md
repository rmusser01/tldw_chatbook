---
id: TASK-38715
title: "File Notes Save failed and Git commit error stay under 4.5:1 in litestep and the ANSI themes"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- accessibility
- theme
- library
priority: low
references:
- tldw_chatbook/css/components/_agentic_terminal.tcss
- tldw_chatbook/css/Themes/themes.py
- Tests/UI/test_library_file_notes_workspace.py
dependencies:
- TASK-36528
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down (TASK-36528's measurements). Checked on dev c2cdb6b82e, 2026-11-03.

TASK-36528 changed the shared rule for the File Notes "Save failed" status and the Git commit error (`.file-notes-git-commit-error, #file-notes-save-status.-error`, `css/components/_agentic_terminal.tcss:454-461`). It now paints `$text` on `$error 14%`. That pair passed 4.5:1 in textual-dark (12.98:1), textual-light (10.90:1) and high_contrast_yellow_black (14.58:1), and in 88 of the 91 shipped themes. Three still fall short:
- **litestep** (`css/Themes/themes.py:2167`): 3.54:1.
- **ansi-dark** and **ansi-light**: 1.92:1. These Textual built-ins use ANSI palette entries such as `color(1)`, which the terminal maps to real colours, so the computed ratio is nominal.

Both are error states that ask the user to act, and the text is the only signal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In litestep, "Save failed" and the Git commit error paint at 4.5:1 or better
- [ ] #2 In ansi-dark and ansi-light the pair reaches 4.5:1, or the task notes record why a ratio computed from ANSI palette entries does not describe what the user sees
- [ ] #3 No theme that passes 4.5:1 today falls below it, and test_high_stakes_file_notes_states_are_legible_in_shipped_themes still passes
<!-- AC:END -->
