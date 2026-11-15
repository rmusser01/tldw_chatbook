---
id: TASK-41057
title: "Console Behavior's field guide spills one row below the Settings inspector at 211x44"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- settings
- ui
- tests
priority: low
references:
- Tests/UI/test_settings_field_guide_in_view.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_settings_field_guide_in_view.py::test_console_behavior_fallback_guide_stays_in_the_inspector_view` fails both of its params on dev ecad404bd4 (2026-11-15) (bootstrap profile). At 211x44, after a click on `settings-console-default-streaming`, or after resizing from 235x52 to 211x44 with `settings-console-default-temperature` focused, the guide's "Validation" row sits at rows 39-40, outside the inspector view (rows 17-39).

TASK-1378.2's report recorded both params red at its base on 2026-11-07. 211x44 is the owner's ThinkPad at full screen, so the last guide row is cut off on a real machine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 211x44 and 235x52, every guide row of the focused Console Behavior field is inside the inspector view
- [ ] #2 Both params pass on dev
<!-- AC:END -->
