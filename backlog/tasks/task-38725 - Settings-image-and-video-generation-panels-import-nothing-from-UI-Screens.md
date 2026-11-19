---
id: TASK-38725
title: "Settings image and video generation panels import nothing from UI/Screens"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- architecture-review-2026-10-25
- settings
priority: low
references:
- tldw_chatbook/Widgets/settings_image_gen_panel.py
- tldw_chatbook/Widgets/settings_video_gen_panel.py
dependencies:
- TASK-36944.8
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36944.8 removed every module-level import of UI/Screens from Widgets. `settings_image_gen_panel.py` and `settings_video_gen_panel.py` still do `from tldw_chatbook.UI.Screens import settings_screen as _owner` inside a function, which is the same upward coupling at call time. AC#1 of 36944.8 covered only module-level imports, so these two stayed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Neither panel imports tldw_chatbook.UI.Screens at any level; what it needs from the screen is passed in or moved to a non-Screens module
- [ ] #2 The Widgets/UI.Screens guard in Tests/Architecture/test_layer_direction.py covers function-level imports, or the task notes say why it does not
- [ ] #3 The image and video generation settings tests show no new failures against dev
<!-- AC:END -->
