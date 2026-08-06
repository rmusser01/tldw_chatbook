---
id: TASK-2610
title: Lab ▸ Speech mode crashes the app with DuplicateIds on mount
status: To Do
assignee: []
created_date: '2026-08-06 18:00'
labels:
  - speech
  - crash
  - lab
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Navigating to Lab ▸ Speech crashes the running app with a Textual
`DuplicateIds: ... 'lab-speech-row-playground'` exception. Reproduced 100% of the time
during slice-3 live verification, and confirmed present on `origin/dev` at `a17f9d369`
(base of `feat/voice-profiles-slice3`) — this is NOT introduced by the voice-profiles
work.

The rail row ids built in `STTSScreen.compose_lab_rail`
(`tldw_chatbook/UI/Screens/stts_screen.py:202-219`) are unique per view key within a
single compose, so a duplicate-id collision means the screen (or its rail) is being
composed/mounted twice rather than that the ids themselves are wrong. The same collision
is the long-standing failure in
`Tests/UI/test_settings_speech_tts_panel.py::test_production_settings_actions_cross_the_pushed_screen_boundary`,
which has been dismissed as an unrelated pre-existing test failure across many tasks —
live driving showed it is not merely a test artifact but a real, user-facing crash.

Impact: the Speech Lab (TTS Playground, Voice Profiles, Studio preferences,
AudioBook/Podcast, Voice Cloning, Speech Recognition) is unreachable through the UI.
Voice profiles can still be created and assigned through other surfaces, but the
playground's "Save result as profile" — the primary creation path — is behind this crash.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Navigating to Lab ▸ Speech mounts the Speech screen without raising, in the running app
- [ ] #2 The double-mount root cause is identified and fixed at its source (not by making the ids unique per mount, which would hide a duplicated mount)
- [ ] #3 `Tests/UI/test_settings_speech_tts_panel.py::test_production_settings_actions_cross_the_pushed_screen_boundary` passes
- [ ] #4 A regression test covers the navigation path that reproduces the crash
<!-- AC:END -->
