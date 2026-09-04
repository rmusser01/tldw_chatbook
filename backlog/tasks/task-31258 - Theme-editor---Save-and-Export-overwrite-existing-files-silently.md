---
id: TASK-31258
title: Theme editor - Save and Export overwrite existing files silently
status: Done
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
updated_date: 2026-09-04 06:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Save writes <name>.toml with no check that a saved theme of that name already exists; Export writes ~/Downloads/<name>_theme.toml on a single Enter with no location choice and no overwrite check. Both are one keypress away from destroying a previous version. Delete already confirms via ConfirmationDialog; reuse it. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Saving over an existing user theme file asks for confirmation naming the file, with an explicit keep option
- [x] #2 Exporting to a path that already exists asks for confirmation; the toast names the written path
- [x] #3 Tests cover confirm and cancel for both
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Save over a different saved theme and Export onto an existing file raise ConfirmationDialog ('Overwrite' / 'Keep existing'); re-saving the theme loaded from that file stays a plain update (_loaded_user_theme tracking). Tests cover confirm-cancel for Save and Export and the no-dialog update path.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
