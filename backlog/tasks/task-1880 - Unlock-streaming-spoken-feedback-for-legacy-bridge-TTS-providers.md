---
id: TASK-1880
title: 'Unlock streaming spoken feedback for legacy-bridge TTS providers (3-leg package)'
status: Done
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
- [x] #1 Legacy-bridge responses carry an honest `sample_rate` (and channels) when the backend knows it.
- [x] #2 Spoken-feedback requests ask for `pcm` only via a caller-scoped override (no change for other TTS callers), through the `synthesize_default`/`_build_request` seam.
- [x] #3 A raw-pcm response that misses sink eligibility is WAV-wrapped before the legacy file path plays it (never an unplayable `.pcm` artifact).
- [ ] #4 openai and kokoro spoken feedback stream through the sink end to end, with the existing fallback tests still green.
<!-- AC:END -->

## Closeout audit 2026-10-11

Verdict: DONE (verified against origin/dev @ a3c47c6c72).

Evidence:
- AC1: `tldw_chatbook/TTS/legacy_bridge.py:844-865` — `synthesize` now carries `sample_rate=24_000` plus `metadata = {sample_rate, channels: 1, sample_encoding: pcm_s16le}` for pcm responses, with the custom-OpenAI-compatible-endpoint honesty carve-out.
- AC2: `tldw_chatbook/TTS/request_admission.py:267,297-306,444` — caller-scoped `response_format_override="prefer_pcm"` drives a `("pcm", "wav")` preference order; the hands-free speech path declares PCM-or-WAV (`synthesize_declared_pcm_or_wav`).
- AC3: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:2329-2333` — raw-pcm responses that miss sink eligibility get a WAV copy via `create_pcm16_wav_copy` before the legacy file path plays them.
- Landed with the speculative duplex voice tree (PR #2504, merge `1f3184655b`, integration commit `b7dd8e53f2` which introduced `prefer_pcm`).
- AC1-AC3 ticked. AC4 left unticked: `Tests/TTS_Events/test_spoken_feedback_streaming.py` exercises openai end-to-end, but no kokoro-specific sink e2e pin was located (kokoro rides the same retained-backend pcm normalization, and `Tests/TTS/test_console_speak_autoplay.py:2549` covers kokoro routing).
