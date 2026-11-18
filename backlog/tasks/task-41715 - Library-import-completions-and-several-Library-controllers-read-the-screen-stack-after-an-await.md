---
id: TASK-41715
title: "Library import completions and several Library controllers read the screen stack after an await"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- ui
- library
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41702's review (round 16). TASK-41702 guarded Home's and Library's footer reads of `app.screen` once a screen is detached at shutdown. The same `self.is_mounted and self.app.screen ...` shape is in five Library import completions (library_screen.py ~26270, 26299, 26324, 27827, 28024) and in controller sites (library_artifacts_controller.py ~206/992, library_artifacts_share_controller.py ~89, library_artifacts_navigation.py ~99, library_navigation_controller.py ~144, library_ingest_controller.py ~2221, library_note_autosave.py ~226, library_prompts_controller.py ~1535/3033). Each can raise `ScreenStackError` if it completes after the stack empties. Some tests reach these through SimpleNamespace fakes (`Tests/Skills/test_skills_import.py`, `Tests/UI/test_library_canvas_scoped_sync.py`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 None of these sites raises once its screen is detached after the exit fence, and each still applies its result while its screen is attached and current
- [ ] #2 A test steps at least one completion past the fence and is red before the change
- [ ] #3 The fakes the change touches are updated
<!-- AC:END -->
