---
id: TASK-42005
title: "A Tool Profiles removal-confirmation test is red on dev and on no lane"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
- settings
priority: low
references:
- Tests/UI/test_settings_tool_profiles.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41526 (round 17). `test_settings_tool_profiles.py::test_removal_confirmation_does_not_interpolate_the_profile_id` fails with `assert '' == 'Remove cancelled'` (~1165). TASK-36387 recorded it as red at dev but owns no fix, and the file is on no PR lane. The product copy it quotes exists (`settings_screen.py` ~6466-6477); the test patches `host.push_screen_wait`, so this looks like test drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is named (test drift or product defect) and fixed
- [ ] #2 The file runs on a PR lane
<!-- AC:END -->
