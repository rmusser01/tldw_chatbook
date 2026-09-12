---
id: TASK-32027
title: Repair Kokoro language selection in Speech Lab generation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 04:56'
updated_date: '2026-09-08 15:08'
labels:
  - tts
  - bug
dependencies: []
references:
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Local Kokoro generation in Speech Lab submits a language placeholder and fails before audio generation. Restore usable language selection and reliable repeated generation across provider changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh Kokoro Playground can generate with automatic voice-based language inference using either engine.
- [x] #2 Explicit language choices submit canonical language codes and survive a provider round trip.
- [x] #3 Loading or unavailable UI values never become provider language options.
- [x] #4 Mounted Playground tests exercise real request admission, repeated attempts, and unchanged Console default admission.
- [x] #5 A fresh Kokoro Playground selects the same ONNX engine as automatic replies; explicitly selecting PyTorch still routes that engine.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: Routine UI/request-construction repair under existing per-request settings and provider ownership contracts; no new boundary.

1. Promote the mounted reproduction to failing behavior tests for automatic and explicit Kokoro language, provider switches, both engines, and repeated attempts.
2. Populate canonical language choices with a voice-inference default, preserve valid session choices, and omit placeholders at request construction.
3. Verify the real admission path, Console control path, affected UI tests and static checks; update TTS guidance and record findings.
4. Live playback follow-up: reproduce the fresh engine mismatch without setting the switch in the harness; default fresh Kokoro controls to ONNX while preserving explicit PyTorch selection, then validate real local synthesis and playback from the integrated dev tree.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kokoro Playground now populates canonical language choices, omits Automatic and placeholders from requests, and preserves explicit language selections across provider switches. Fresh controls select ONNX, matching automatic replies; explicitly switching it off still selects PyTorch. These are request/session choices under ADR-039, with no new stored engine preference.

Mounted regressions cross the real registry, resolver, service, event handler and artifact handling. The language tests and two fresh-engine cases failed before their respective fixes; all nine now pass. The integrated dev TTS/UI/admission gate passed 313 tests. Real local ONNX inference subsequently generated and played a fresh Speech Lab sentence and two consecutive Speak replies requests for each of WAV and MP3. All six files decoded to complete 5.632, 6.357 and 5.717-second sentences, and independent local transcription recovered their expected content. WAV Console playback reported two SinkDrained events; Lab and MP3 file playback completed through afplay with exit 0. No real LLM request was needed: synthetic completed assistant rows exercised the actual snapshot/admission/TTS/playback path.

Playback validation also found non-PCM chunk encoding truncation, repaired in TASK-32027.1. The optional PyTorch path has real-codec regression coverage with inference replaced; live neural inference/playback was ONNX only. Updated speech_param_group.py, Playground model/pane/synthesis code, mounted tests, TTS guide, Console guidance and the incident-based testing lesson. Independent review, changed-file formatting, zero introduced baseline-relative Ruff findings and all six derived preflight checks pass. Python 3.12.11/SQLite 3.49.1 differ from the reporter's environment, so the reporter's unspecified retry/selection trigger remains unconfirmed.

Live evidence: /private/tmp/kokoro-playback-validation/{evidence.json,mp3/evidence.json,transcription-evidence.json}; harness /private/tmp/validate-kokoro-playback.py. Targeted test log: /private/tmp/kokoro-pr-tts-gate.log. The complete fix and newer Console integration are isolated on codex/kokoro-speech-trace-recovery; unrelated original-workspace changes are preserved.
<!-- SECTION:NOTES:END -->
