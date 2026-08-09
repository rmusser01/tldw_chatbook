---
id: TASK-3797
title: >-
  Settings Appearance tests fail: _appearance_bool_label undefined (9 call
  sites)
status: To Do
assignee: []
created_date: '2026-08-09 16:47'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three Appearance-category tests in Tests/UI/test_settings_configuration_hub.py fail on the current dev baseline, unrelated to any change in this branch. SettingsScreen calls self._appearance_bool_label(...) at 9 sites (Animations/Smooth scrolling/Reduce motion/ASCII glyphs button labels and their sync logic) but the method is never defined anywhere in settings_screen.py, raising AttributeError. Confirmed pre-existing during the supervisor-fleet PR-1 program: reproduced in isolation on this branch's HEAD before task-6's own edits touched anything (git show confirmed identical on the pre-task-6 commit).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_settings_appearance_renders_guided_defaults_and_validates passes
- [ ] #2 test_settings_appearance_revert_restores_loaded_values passes
- [ ] #3 test_settings_appearance_preview_updates_runtime_without_saving passes
- [ ] #4 _appearance_bool_label is defined (or every call site replaced with the correct existing helper) and the Appearance category's Animations/Smooth scrolling/Reduce motion/ASCII glyphs buttons render correct Enabled/Disabled labels in a live run
<!-- AC:END -->
