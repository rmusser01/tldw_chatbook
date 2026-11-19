---
id: TASK-39104
title: 'test_play_button_missing_file_warns_through_the_validator is order-dependent'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - tests
dependencies: []
references:
  - Tests/UI/test_artifacts_screen_reports.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_artifacts_screen_reports.py::test_play_button_missing_file_warns_through_the_validator` fails after `test_play_button_validates_path_and_plays_normalized_result`, which leaves `script-1-audio-1.wav` in the shared audio folder, so the "missing file" is present. It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), seen as an order-dependent failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two tests pass in either order and in the full file
- [ ] #2 The first test no longer leaves script-1-audio-1.wav behind, or the second uses a name that cannot exist
<!-- AC:END -->
