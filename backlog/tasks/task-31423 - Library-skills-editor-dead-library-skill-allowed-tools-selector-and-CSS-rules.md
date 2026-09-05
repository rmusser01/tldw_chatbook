---
id: TASK-31423
title: >-
  Library skills editor: dead #library-skill-allowed-tools selector and CSS
  rules
status: To Do
assignee: []
created_date: '2026-09-05 01:07'
labels:
  - library
  - skills
  - tech-debt
  - css
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library skills editor's tool-selection UI moved from a plain Input to a SelectionList-based chooser (see library_skills_canvas.py's _tool_picker_selections/skill_allowed_tools_sequence), but the old #library-skill-allowed-tools Input id was never fully retired. Two @on(Input.Changed, "#library-skill-allowed-tools") handlers still exist (library_skills_controller.py and its screen-side delegator in library_screen.py) and never fire because no widget with that id is ever composed anywhere in the editor, and matching dead CSS rules remain in tldw_chatbook/css/screen_agentic_library.tcss (lines ~1424, ~1439) and tldw_chatbook/css/components/_agentic_terminal.tcss (lines ~2591, ~2606). Found during the wave-4 Library skills decomposition final review; out of scope for that review to fix, filed here so the dead surface is removed as its own atomic change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dead #library-skill-allowed-tools @on(Input.Changed, ...) handler(s) and any now-unreachable code they alone guard are removed from library_skills_controller.py and library_screen.py, with the full Library skills test suites still green
- [ ] #2 The dead #library-skill-allowed-tools CSS rules are removed from screen_agentic_library.tcss and _agentic_terminal.tcss, and the CSS bundle sync check in scripts/preflight.sh stays green
<!-- AC:END -->
