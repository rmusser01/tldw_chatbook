---
id: TASK-15422
title: >-
  TTS model-route failure surfaces as "Unexpected TTS generation failure" and
  fires twice per speak
status: To Do
assignee: []
created_date: '2026-08-11 12:00'
labels:
  - tts
  - speech
  - ux
  - uat
priority: medium
dependencies:
  - TASK-15420
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT on `origin/dev` `82b595049` (2026-08-11), companion to TASK-15420.

When Console speak fails on `UnknownLegacyModelError` (a deterministic
configuration/selection problem):

1. **The toast is misleading.** `UnknownLegacyModelError` subclasses
   `LookupError`, so `_tts_outcome_code` buckets it as `generation_failed` and
   `_tts_error_copy` falls through to "Unexpected TTS generation failure;
   retry" (`Event_Handlers/TTS_Events/tts_events.py`). Retrying can never help;
   the actionable copy family ("TTS is not configured; open STTS Settings")
   exists but is not reached. A user who followed the OpenAI-compatible-server
   doc gets no hint that their model name is what was rejected.
2. **One 🔊 click logs the failure twice.** Every observed click produced two
   back-to-back `ERROR TTS generation failed (outcome_code=generation_failed)`
   entries with the same timestamp (4 clicks → 8 errors across two app
   sessions), implying two generation attempts or a double-reported failure
   path per click — worth tracing regardless of the copy fix, since it doubles
   metric counts (`tts_generation_total`).
3. Once, in the first session, the message's speak control stayed on
   "⏹ Stop speech" for minutes after the instant failure (fresh selection of
   the same message still showed ⏹ ~4 minutes later); in a later controlled
   repro the control recovered within ~10s. Intermittent, lower priority, but
   noted since a stuck ⏹ hides the retry affordance entirely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A model/selection rejection surfaces copy that points at the TTS model configuration rather than a generic retry
- [ ] One speak action produces exactly one generation attempt and one failure log/metric on failure
- [ ] After a synchronous failure the message speak control returns to 🔊 promptly and deterministically
<!-- AC:END -->
