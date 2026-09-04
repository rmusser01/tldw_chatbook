---
id: TASK-31261
title: Theme editor polish - no-op Reset toast, three primary buttons, group naming,
  lowercase hex
status: Done
created_date: 2026-09-04 05:24
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: low
updated_date: 2026-09-04 06:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small consistency items from the review: Reset toasts 'Theme reset to original values' when nothing changed; New, Apply and Generate are all variant=primary; the tree calls the 58 shipped themes 'Custom Themes' and prefixes the user's own with 'user:'; hex case differs between paths. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reset with no changes gives a neutral message or no toast; Reset with changes still confirms (pinned behaviour)
- [x] #2 Exactly one primary-variant button per action row (Apply primary; New/Generate default)
- [x] #3 Tree groups read 'Your themes', 'Shipped themes', 'Built-in'; leaves have no user: prefix and the delete/shadowing tests still pass with updated labels
- [x] #4 All hex values shown are uppercase
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reset with no edits now says 'No changes to reset' (still no dialog; pinned test updated); Apply is the only primary-variant button; tree groups renamed and 'user:' prefix dropped (done in TASK-31256); generated hex uppercased (TASK-31253).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
