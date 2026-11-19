---
id: TASK-39106
title: 'The Watchlists audio row checks one path and plays another'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - security
  - watchlists
dependencies: []
references:
  - tldw_chatbook/UI/Screens/watchlists_collections_screen.py
  - tldw_chatbook/UI/Library_Modules/library_artifacts_controller.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Watchlists audio row has the same confinement gap that TASK-39040 fixed for Artifacts (TASK-39040 and its fix are on `origin/fix/windows-audio-fallback-no-cmd-exe`, not yet on dev, so the line reference below is to that branch): a relative row is checked against `briefing_audio_dir()` but opened relative to the cwd (`watchlists_collections_screen.py:11507-11517`). It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), read on dev. The fix pattern is `library_artifacts_controller.py:1053`: `validate_path(..., redact_paths=True)`, then play the resolved path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Watchlists audio row plays the same resolved path it validated, so a relative row cannot escape briefing_audio_dir()
- [ ] #2 A test pins a relative row that resolves differently against the cwd and the audio dir
<!-- AC:END -->
