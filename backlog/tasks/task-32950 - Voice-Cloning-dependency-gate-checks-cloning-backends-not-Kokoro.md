---
id: TASK-32950
title: Voice Cloning dependency gate checks cloning backends, not Kokoro
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-25 08:10'
updated_date: '2026-09-25 08:40'
labels:
  - tts
  - voice-cloning
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Opening Lab ▸ Speech ▸ Voice Cloning showed "Feature Not Available: Text-to-Speech — kokoro-onnx and pyaudio are not installed" whenever Kokoro's local TTS was absent, even with a working cloning backend (OmniVoice, Higgs, Chatterbox) installed. None of the cloning backends use Kokoro or pyaudio, so users were told to install the wrong packages, and the window then opened on Higgs even when Higgs was not installed. Found during PR #2825 (OmniVoice) UAT.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With any cloning backend installed, opening Voice Cloning shows no dependency alert
- [x] #2 With no cloning backend installed, the alert names the voice-cloning backends and their pip extras, not Kokoro
- [x] #3 The window opens on an installed backend (Higgs stays the default when installed)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The window gated on `DEPENDENCIES_AVAILABLE["tts_processing"]` (Kokoro local TTS). It now asks `speech_local_dependency_availability(refresh=True)` (non-importing find_spec probes, the same facts the Speech Lab's local-capability rows use) whether OmniVoice, Higgs or Chatterbox is present; only if none is does it show `alert_voice_cloning_not_available`, which replaces the Kokoro-specific `alert_tts_not_available` (its only caller) and recommends `omnivoice_tts` (CPU/ONNX, no torch) while naming `higgs_tts` and `chatterbox`. When Higgs is missing the backend select defaults to the first installed backend.

Tests (Tests/UI/test_voice_cloning_window_omnivoice.py): no alert with only OmniVoice installed (fails against the old gate — negative control run), correct alert text when none is installed, OmniVoice-only default; the Higgs-default test was made hermetic. Live: scratch profile without Kokoro/pyaudio opens straight to Voice Cloning on OmniVoice with the saved profile listed.

Files: tldw_chatbook/UI/Voice_Cloning_Window.py, tldw_chatbook/Utils/widget_helpers.py, Tests/UI/test_voice_cloning_window_omnivoice.py.
<!-- SECTION:NOTES:END -->
