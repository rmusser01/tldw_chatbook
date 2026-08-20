---
id: TASK-18933
title: 'Voice mode: on-device wake words'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - voice
  - stt
  - privacy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add open-vocabulary wake-word detection to chatbook's existing voice mode (Mic dictation and TTS with barge-in already shipped), porting hermes-agent's wake-word feature (2026-08-19 hermes-release review) under chatbook's local-first constraints. A user-chosen phrase (e.g. "hey chatbook") arms listening; a spoken "stop" (configurable) ends the voice session hands-free. Detection must run fully on-device — no audio leaves the machine while waiting for the wake phrase — consistent with the local-private-data boundary (ADR-029). Multiple phrases are optional. Opt-in, OFF by default: when off, microphone behavior is byte-identical to today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User-configurable wake phrase(s); detection runs entirely on-device — pinned by a test asserting zero network egress while armed
- [ ] #2 A detected wake phrase arms the existing dictation path (no parallel audio pipeline); a spoken stop-phrase ends the voice session hands-free on every voice-capable surface in the app
- [ ] #3 OFF by default; with the feature off, no microphone access or audio path changes versus today (tested)
- [ ] #4 The detection engine choice, its dependency footprint, latency, and CPU budget are documented; false-positive/false-negative rates are measured on a recorded test set and reported honestly
- [ ] #5 Tests cover config gating, arm/disarm flow, the no-egress guarantee, stop-phrase handling, and failure behavior when the engine or microphone is unavailable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/074-voice-wake-word-on-device-boundary.md (to be drafted before implementation).
Reason: introduces an always-on audio dependency and a new privacy boundary (on-device detection only, no egress while armed) — dependency/tooling choice plus security/privacy decision per the ADR policy.

1. Draft ADR-074: engine choice (e.g. open-wake-word-class dependency vs vendored detector), arm/disarm lifecycle, privacy guarantees
2. Wake listener lifecycle integrated with the existing voice-mode audio path
3. Phrase config + stop-phrase handling
4. No-egress test, gating tests, measured accuracy report, docs (Speech services guide)
<!-- SECTION:PLAN:END -->
