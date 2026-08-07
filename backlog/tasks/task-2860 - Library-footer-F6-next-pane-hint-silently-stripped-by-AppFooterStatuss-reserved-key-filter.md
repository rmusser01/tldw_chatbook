---
id: TASK-2860
title: >-
  Library footer 'F6 next pane' hint silently stripped by AppFooterStatus's
  reserved-key filter
status: To Do
assignee: []
created_date: '2026-08-07 07:30'
labels:
  - library
  - footer
  - keyboard
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
AppFooterStatus.set_shortcut_context filters any workbench-shortcut hint whose key is in _RESERVED_GLOBAL_KEYS = {f1, f6, ctrl+p, ctrl+q} (tldw_chatbook/Widgets/AppFooterStatus.py). LibraryScreen's LIBRARY_LANDING_SHORTCUTS, LIBRARY_GENERAL_SHORTCUTS, LIBRARY_NOTES_FILES_SHORTCUTS, and (as of task-2856) LIBRARY_DETAIL_BACK_SHORTCUTS/LIBRARY_LIST_SHORTCUTS all advertise ("F6", "next pane") for the screen's own workbench pane-cycle action (action_focus_next_workbench_pane), but that hint is silently dropped from the rendered footer text and replaced by the global GLOBAL_HINTS suffix (which shows an UNRELATED "F6 panes" hint for a different, app-level F6 action). Discovered while re-running the full Tests/UI/test_library_shell.py suite for task-2856: test_landing_footer_advertises_the_landing_keyboard_story deterministically fails at HEAD (confirmed via a direct A/B: still fails with task-2856's own footer-registration change fully reverted), so this predates task-2856 and is not caused by it. The F6 KEY ITSELF still works (Textual resolves the binding regardless of footer text); only the per-screen, action-specific hint text is wrong/missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Library landing footer's rendered shortcut_text literally includes the screen's own F6 hint copy (e.g. 'next pane'), not just the unrelated global 'F6 panes' hint
- [ ] #2 test_landing_footer_advertises_the_landing_keyboard_story (Tests/UI/test_library_shell.py) passes
- [ ] #3 Audit other screens registering an F6 (or F1/Ctrl+P/Ctrl+Q) workbench shortcut hint through AppFooterStatus for the same silent-drop, and fix or document them
<!-- AC:END -->
