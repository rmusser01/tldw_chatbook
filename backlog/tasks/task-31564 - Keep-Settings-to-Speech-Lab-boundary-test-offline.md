---
id: TASK-31564
title: Keep Settings-to-Speech-Lab boundary test offline
status: Done
assignee: []
created_date: '2026-09-05 02:25'
updated_date: '2026-09-05 02:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the production Settings navigation boundary test exercises screen handoff without probing a real local TTS endpoint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The boundary test replaces the Speech playground TTS service seam with a deterministic fake.
- [x] #2 The test still proves Settings save and Speech Lab navigation cross the production App boundary.
- [x] #3 The focused test passes without blocked network attempts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the network guard failure. 2. Patch the existing SpeechPlaygroundPane service seam with the shared deterministic fake. 3. Run the focused test and the speech settings module, then record verification. ADR required: no. ADR path: N/A. Reason: This is isolated test-harness correction and does not change application architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused the existing `FakeTTSService` at the `SpeechPlaygroundPane._tts_service_factory` seam so navigation into Speech Lab still exercises the production App and screen boundary without contacting the default local audio.cpp endpoint. The focused boundary test passes with the network guard enabled, the complete Speech/TTS settings module passes 123/123, Ruff passes for the modified test module, and `git diff --check` is clean. Modified `Tests/UI/test_settings_speech_tts_panel.py`. ADR required: no.
<!-- SECTION:NOTES:END -->
