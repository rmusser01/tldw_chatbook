---
id: TASK-1880
title: 'Unlock streaming spoken feedback for legacy-bridge TTS providers (3-leg package)'
status: To Do
assignee: []
created_date: '2026-08-02 12:00'
labels: [tts, audio, streaming]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The streaming PCM sink (V3 phase 1) serves only the audio.cpp adapter today: the legacy
backend bridge (`TTS/legacy_bridge.py` ~:692) constructs `TTSAudioResponse` without `sample_rate`,
so `sink_plan()` returns None for every legacy provider (openai, kokoro, ...). The Task-4 review
adjudicated this deferral and identified that implementing the naive request-side pcm ask alone
would REGRESS openai feedback (raw `.pcm` artifact the file player cannot play → silent). All
three legs are needed together. Origin: streaming-sink final review M1/M3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Legacy-bridge responses carry an honest `sample_rate` (and channels) when the backend knows it.
- [ ] #2 Spoken-feedback requests ask for `pcm` only via a caller-scoped override (no change for other TTS callers), through the `synthesize_default`/`_build_request` seam.
- [ ] #3 A raw-pcm response that misses sink eligibility is WAV-wrapped before the legacy file path plays it (never an unplayable `.pcm` artifact).
- [ ] #4 openai and kokoro spoken feedback stream through the sink end to end, with the existing fallback tests still green.
<!-- AC:END -->
