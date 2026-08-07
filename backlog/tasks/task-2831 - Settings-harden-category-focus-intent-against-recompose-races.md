---
id: TASK-2831
title: Settings harden category focus intent against recompose races
status: To Do
assignee: []
created_date: '2026-08-05 00:20'
labels:
  - settings
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code review follow-up to TASK-1338: the one-shot _pending_category_focus_value is consumed by ANY recompose, and recomposes are not atomic - a mount-time sync-rows recompose in flight can consume the intent before the category recompose destroys the focused widget (same failure class as task-1338, narrow timing window). Switch to the consume-on-satisfaction pattern already used by _pending_navigation_focus_selector (settings_screen.py:8020-8029), or re-assert via call_after_refresh verifier. Also from the same review: hidden-button focus edge when a search filter hides the selected category (:7863-7867), add docstring on recompose() override (:8012), narrow bare except in test_settings_category_sweep.py:40-41, note stale echo clobber limitation in settings_theme_editor.py:393-411.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Focus intent is only cleared when the intended focus is actually satisfied,Pilot test covers category click during the post-mount sync recompose window,Review polish items 2-5 from the task-1338 quality review are addressed or documented
<!-- AC:END -->
