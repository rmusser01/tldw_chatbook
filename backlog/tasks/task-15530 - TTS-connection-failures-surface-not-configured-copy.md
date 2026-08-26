---
id: TASK-15530
title: >-
  TTS connection failures surface "not configured" copy instead of a
  connection outcome
status: Done
assignee: []
created_date: '2026-08-11 16:20'
labels:
  - tts
  - speech
  - ux
  - uat
priority: medium
dependencies:
  - TASK-15422
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed during the TASK-15422 live verification (dead-port custom Base URL,
Console 🔊): the failure toast read **"TTS is not configured; open STTS
Settings"** and the metric outcome was `configuration_invalid` — for a
*reachability* failure against a fully configured endpoint. Mechanism:
`OpenAITTSBackend.generate_speech_stream` wraps `httpx.RequestError` in a
plain `ValueError("Unable to connect to TTS service...")`, and
`tts_events`' `_tts_outcome_code`/`_tts_error_copy` bucket every
`ValueError` as configuration-invalid. A user whose server is simply down
(or whose port is wrong) is told their configuration is missing, which
points them away from the actual fix.

The bounded `TTSOperationCode` set already has `connection_unavailable`.
The backends' shared contract is "failures are ValueError" (consumers in
audiobook, media reading, and briefings catch ValueError), so the typed
error must remain a ValueError subclass to keep every existing consumer
working unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] A connection failure to the TTS endpoint surfaces copy about reaching the server (naming the Base URL as the thing to check), not "not configured"
- [x] The metric outcome for a connection failure is `connection_unavailable`, not `configuration_invalid`
- [x] Existing consumers that catch `ValueError` from backend streams are unaffected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: typed-error contract (ValueError subclass), outcome mapping, copy
   mapping, and a backend test pinning the network branch raises the type.
2. GREEN: `TTSBackendConnectionError(ValueError)` in `base_backends.py`;
   openai backend's `httpx.RequestError` branch raises it; `tts_events`
   maps it to `connection_unavailable` + reachability copy BEFORE the
   ValueError bucket it subclasses.
3. Live dead-port verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`TTSBackendConnectionError(ValueError)` added to `TTS/base_backends.py` —
a ValueError subclass on purpose, so every existing backend-stream
consumer catching ValueError (audiobook, media reading, briefings) is
unchanged; pinned by an explicit subclass test. The openai backend's
`httpx.RequestError` branch raises it (same message); `tts_events` maps
it to the bounded `connection_unavailable` outcome with copy "Unable to
reach the TTS server; check that it is running and the Base URL in STTS
Settings", ordered before the generic ValueError bucket. Scope is the
openai backend (the custom-endpoint provider this arc is about); other
API backends keep their existing wrapping. Live-verified dead-port: the
toast now names reachability and the Base URL instead of "not
configured". 122 neighboring consumer tests green.
<!-- SECTION:NOTES:END -->
