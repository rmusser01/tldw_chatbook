---
id: TASK-33061
title: Revert never writes a missing launch default back to config
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. revert_theme persists previous_launch_default unconditionally, so reverting after a launch-default-missing start rewrites the broken name to config and the 'Launch default missing' notice returns. The Revert chip also lingers when it would change nothing. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Revert leaves the saved launch default untouched when the previous one is not an available theme, and its label says so
- [x] #2 The Revert chip is hidden when reverting would change neither the active theme nor the launch default
- [x] #3 Existing Try/Use/Revert tests are updated to the new contract and pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. revert_theme: skip the launch-default write when previous_launch_default is not a registered theme (signature/return unchanged).
2. Picker label: '(launch unchanged)' in that case.
3. Picker hides the chip when the target active theme is already active and the launch default would not change.
4. Tests: catalog unit test + picker tests; update the two pinning tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
revert_theme (theme_catalog.py) skips the launch-default write when previous_launch_default is not in app.available_themes (new helper launch_default_restorable); signature and (restored, caches_reloaded) return unchanged -- that case returns (True, True). The picker's _sync_revert_chip labels it '(launch unchanged)' and hides the chip (and its row) when the target is already active and the launch default would not change; the pending change is kept so a later Try/Use still merges into it. The two named pinning tests already held under the new contract and pass unchanged; test_revert_label_strips_control_characters_from_the_launch_default was updated (its hostile launch default is unregistered, so the label no longer names it). New: test_revert_never_writes_a_missing_launch_default_back (catalog), test_revert_leaves_a_missing_launch_default_alone and test_revert_chip_hides_when_it_would_change_nothing (picker).
<!-- SECTION:NOTES:END -->
