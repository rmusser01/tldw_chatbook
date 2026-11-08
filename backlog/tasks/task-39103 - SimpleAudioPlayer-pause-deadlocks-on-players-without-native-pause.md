---
id: TASK-39103
title: 'SimpleAudioPlayer.pause deadlocks on players without native pause'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - tts
  - bug
dependencies: []
references:
  - tldw_chatbook/TTS/audio_player.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`SimpleAudioPlayer.pause()` deadlocks on players without native pause, such as afplay: it re-takes its own non-reentrant lock. It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), read on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 pause() returns on a player without native pause (afplay) and playback state is consistent afterwards
- [ ] #2 A test pauses such a player with a timeout and fails on the deadlock
<!-- AC:END -->
