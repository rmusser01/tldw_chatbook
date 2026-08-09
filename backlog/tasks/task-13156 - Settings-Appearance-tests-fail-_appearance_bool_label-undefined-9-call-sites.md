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
Root cause: the toggle-Button conversion for reduce-motion/ascii-glyphs/smooth-scrolling landed on dev with 9 call sites to _appearance_bool_label but the method was undefined at HEAD. Corrected archaeology (verified with `git show` on each sha): the method DID exist in history -- it was added at 531ef6f67 ("Functionalize Settings appearance defaults", 2026-06-07), then removed together with all of its call sites by b2b7b20f5 ("fix(settings): UX critique fixes — crash, honest hints, layout, IA (tasks 1338-1346)", 2026-08-05). The reduce-motion/ascii-glyphs/smooth-scrolling call sites were reintroduced later by 7dbbc401b ("feat(console): UX review remediation — all 24 findings (TASK-2154)", 2026-08-07), but that commit only restored the *call sites* -- not the `def` -- leaving the dangling AttributeError this task fixes. Restored the helper (Enabled/Disabled from _appearance_setting_values draft-over-loaded precedence, matching the file's existing bool-label conventions at :3395/:3475) and repaired the one stale assertion that still queried smooth-scrolling as a Checkbox (the shipped UI is a Button toggle). The restored `return` statement is byte-identical to the one 531ef6f67 originally shipped; a new docstring was added on top of it. All 7 Appearance hub tests pass.
<!-- SECTION:NOTES:END -->
