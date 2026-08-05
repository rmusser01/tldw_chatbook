---
id: TASK-2452
title: >-
  TTS Playground Save-as-profile is unreachable for legacy providers
  (Studio-effective path never attaches provenance)
status: To Do
assignee: []
created_date: '2026-08-05 04:49'
updated_date: '2026-08-05 04:49'
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
- [ ] #1 Generating audio for any of the seven providers through the real TTS Playground (Generate button, not a test harness calling _generate_legacy directly) leaves the result save-eligible when its provenance would otherwise qualify, matching the audio_cpp behavior
- [ ] #2 _generate_studio_effective attaches a TTSRequestedSelectionSnapshot for the six legacy providers using the same exact-selection provenance _generate_legacy already builds, not only for audio_cpp
- [ ] #3 A regression test drives the real Playground Generate path (not _generate_legacy in isolation) for a legacy provider and asserts the resulting artifact is profile_save_eligible
- [ ] #4 audio_cpp's existing Studio-effective save-as-profile behavior is unchanged (characterized before the fix, pinned after)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Found live, 2026-08-04, during task-2450's Task 6 live verification: a real Generate click against the real OpenAI backend produced a real, successful artifact (confirmed via the app's own RichLog 'TTS generation complete!' and the persistent-diagnostics event log), but 'Save result as profile' stayed hidden/disabled. Root cause isolated by temporary debug instrumentation (added and reverted, never committed) proving current_audio_artifact.requested_selection was None: SpeechPlaygroundPane is always constructed with studio_preferences=load_result.snapshot (UI/STTS_Window.py:4711, never None), so _generate_tts (UI/Speech/speech_synthesis_mixin.py) always builds a non-None studio_draft and every real Generate click's STTSPlaygroundRequest.studio_preferences is set, routing _generate_tts_worker (Event_Handlers/STTS_Events/stts_events.py:1060) into _generate_studio_effective, never _generate_legacy. Confirmed the branch's own diff (git diff ab9105c9d..HEAD -- Event_Handlers/STTS_Events/stts_events.py) touched only _generate_legacy's provenance block (~848-877); _generate_studio_effective (~895-972) was untouched and still hard-codes 'if effective.provider_id == "audio_cpp" else None' at line ~953. Worked around for task-2450's own live verification with an honest, clearly-labeled in-process substitute (real TTSService.generate_audio_stream + real TTSProfileService.create_from_artifact, bypassing only this broken UI glue) -- see task-2450's Task 6 report for the full trace.
<!-- SECTION:NOTES:END -->
