---
id: TASK-31259
title: Theme editor - Live Preview only changes on Apply and previews nothing from
  Chatbook
status: To Do
created_date: 2026-09-04 05:24
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
The preview is three stock buttons and three boxes styled by the app's CSS tokens, so it reflects the applied app theme, not the palette being edited; the User Guide calls it decorative. Users edit blind until Apply. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preview surfaces repaint from the edited palette on every colour change without Apply
- [ ] #2 Preview shows a Chatbook-shaped stub (a few Console transcript rows: user/assistant/tool with rail chrome) using primary, secondary, accent, background, surface, panel, foreground, success, warning and error
- [ ] #3 Docs/User_Guide/settings.md no longer calls the preview decorative
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
