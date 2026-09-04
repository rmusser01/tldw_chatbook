---
id: TASK-31258
title: Theme editor - Save and Export overwrite existing files silently
status: To Do
created_date: 2026-09-04 05:24
dependencies:
- TASK-31251
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Save writes <name>.toml with no check that a saved theme of that name already exists; Export writes ~/Downloads/<name>_theme.toml on a single Enter with no location choice and no overwrite check. Both are one keypress away from destroying a previous version. Delete already confirms via ConfirmationDialog; reuse it. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saving over an existing user theme file asks for confirmation naming the file, with an explicit keep option
- [ ] #2 Exporting to a path that already exists asks for confirmation; the toast names the written path
- [ ] #3 Tests cover confirm and cancel for both
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
