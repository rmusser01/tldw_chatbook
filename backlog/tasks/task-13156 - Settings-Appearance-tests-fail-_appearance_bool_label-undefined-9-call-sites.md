---
id: TASK-13156
title: >-
  Settings Appearance tests fail: _appearance_bool_label undefined (9 call
  sites)
status: Done
assignee: []
created_date: '2026-08-09 16:47'
updated_date: '2026-08-09 18:54'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the toggle-Button conversion for reduce-motion/ascii-glyphs/smooth-scrolling landed on dev with 9 call sites to _appearance_bool_label but the method was never defined (never existed in history; confirmed via git log -S across origin/dev). Restored the helper (Enabled/Disabled from _appearance_setting_values draft-over-loaded precedence, matching the file's existing bool-label conventions at :3395/:3475) and repaired the one stale assertion that still queried smooth-scrolling as a Checkbox (the shipped UI is a Button toggle). All 7 Appearance hub tests pass.
<!-- SECTION:NOTES:END -->
