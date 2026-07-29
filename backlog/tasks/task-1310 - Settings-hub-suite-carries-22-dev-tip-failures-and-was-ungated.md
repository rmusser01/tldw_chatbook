---
id: TASK-1310
title: 'Settings-hub suite carries 22 dev-tip failures and was ungated'
status: To Do
assignee: []
created_date: '2026-07-28 13:30'
labels: [settings, tests, regressions]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-1234's review, the first settings-hub gate on the fleet-findings branch surfaced 22 failing tests in Tests/UI/test_settings_configuration_hub.py present at the dev-tip base (byte-identical name sets at base 93bf5518c and branch HEAD — none caused by the branch): a provider/model-resolution TypeError family, a save_setting_to_cli_config/save_settings_to_cli_config naming-drift family, and a PrivatePathError. The suite was last known green in this program pre-#1050 and none of the recent trains gated it. Fix the regressions and keep the hub in routine verification gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 22 named failures (see TASK-1234 review) pass or are individually dispositioned with root-cause notes.
- [ ] #2 The originating dev commits are identified (naming drift especially).
- [ ] #3 The hub suite is listed in the standard Console-area verification gates going forward.
<!-- AC:END -->
