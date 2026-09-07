---
id: TASK-2452
title: >-
  TTS Playground Save-as-profile is unreachable for legacy providers
  (Studio-effective path never attaches provenance)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 04:49'
updated_date: '2026-08-05 05:42'
labels: []
dependencies:
  - TASK-2450
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of task-2450 (voice profiles slice 1) found that the TTS Playground's 'Save result as profile' button never becomes eligible for any non-audio_cpp provider in the real running app, even though the profile service and _generate_legacy code path were extended to support it. The Playground always mounts with a non-None studio_preferences snapshot, so every real Generate click routes through _generate_studio_effective, not _generate_legacy -- and _generate_studio_effective only attaches a TTSRequestedSelectionSnapshot when the effective provider is audio_cpp (Event_Handlers/STTS_Events/stts_events.py, _generate_studio_effective, ~line 943-955), unconditionally leaving requested_selection None for every legacy provider. _generate_legacy's own provenance attachment (fixed in slice 1) is consequently dead code from the live Playground UI: no user can ever reach it through Generate. This directly contradicts the slice's intended outcome that a legacy-provider generation is save-eligible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generating audio for any of the seven providers through the real TTS Playground (Generate button, not a test harness calling _generate_legacy directly) leaves the result save-eligible when its provenance would otherwise qualify, matching the audio_cpp behavior
- [x] #2 _generate_studio_effective attaches a TTSRequestedSelectionSnapshot for the six legacy providers using the same exact-selection provenance _generate_legacy already builds, not only for audio_cpp
- [x] #3 A regression test drives the real Playground Generate path (not _generate_legacy in isolation) for a legacy provider and asserts the resulting artifact is profile_save_eligible
- [x] #4 audio_cpp's existing Studio-effective save-as-profile behavior is unchanged (characterized before the fix, pinned after)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in-slice (task-6b, controller ruling: this and TASK-2453 defeat the slice's own requirement and are not follow-ups). TDD: added test_studio_generation_attaches_provenance_for_legacy_effective_provider (RED against pre-fix code, confirmed) plus a degrade-gracefully sibling test and a profile_save_eligible pin on the existing audio_cpp studio test, all in Tests/TTS/test_stts_audio_cpp_generation.py. Fix: factored a shared STTSEventHandler._build_requested_selection(...) helper (provider_id, model_id, voice_id, response_format, speed, configuration_revision callable) that both _generate_legacy and _generate_studio_effective now call; _generate_studio_effective's audio_cpp-only ternary is gone -- it calls the helper unconditionally using effective.response_format/effective.speed/effective.revisions.provider_configuration, which research confirmed are valid, already-resolved values for every provider (audio_cpp's effective values are guaranteed wav/1.0 by TTSEffectiveSelectionSnapshot's own validation, so unifying the two providers onto one code path changes nothing for audio_cpp). Mutation-verified twice: removing the helper's try/except makes both degrade-gracefully tests fail with a real exception; reinstating the old audio_cpp-only gate makes the new provenance test fail. Full Tests/TTS/ (2224 passed) and the branch's 12 targeted files (943 passed) stayed green throughout. Live re-verified end to end against a real OpenAI account (fresh tmux session, scratch profile): Generate produced real audio, 'Save result as profile' was visible and clickable, the name modal saved, and the app reported 'Voice profile saved.' -- confirmed present in the Voice Profiles library as Unverified. See task-2450's task-6b report for the full trace.
<!-- SECTION:NOTES:END -->
