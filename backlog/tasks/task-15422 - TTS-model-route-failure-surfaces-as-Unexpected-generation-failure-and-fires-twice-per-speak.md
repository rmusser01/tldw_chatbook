---
id: TASK-15422
title: >-
  TTS model-route failure surfaces as "Unexpected TTS generation failure" and
  fires twice per speak
status: Done
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
- [x] A model/selection rejection surfaces copy that points at the TTS model configuration rather than a generic retry
- [x] One speak action produces exactly one generation attempt and one failure log/metric on failure
- [x] After a synchronous failure the message speak control returns to 🔊 promptly and deterministically
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Root-cause the "double fire" before assuming two generations: the mock
   server received exactly ONE request per successful click, so the doubling
   had to be in the logging pipeline, not dispatch.
2. RED tests per finding; GREEN each: log duplication, error copy mapping,
   Console speak-marker clear on failure.
3. Live verification: dead-port TTS endpoint, one 🔊 click.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three distinct findings, three scoped fixes (TDD red-first for each):

1. **"Two failures per click" was a log-pipeline duplication, not a double
   generation.** `Logging_Config._setup_logging` installs the canonical
   loguru→stdlib forward (`_forward_loguru_to_standard`, TRACE,
   diagnose=False per task-2119) on every boot path, and
   `TldwCli._setup_buffered_logging` then installed a SECOND loguru→stdlib
   sink — every loguru record reached the root logger's
   `PersistentLogHandler` twice, so the Logs screen showed every
   application log line (and counted every error) twice. Fix: the app-level
   bridge is removed with a comment pointing at the canonical forward.
   Test: `Tests/test_logs_buffer_single_record_per_emission.py` replicates
   the production sink layout and pins one emission → one buffered record
   (was 2).

2. **`UnknownLegacyModelError` now maps to `model_invalid`** (already in the
   bounded `TTSOperationCode` set) with copy "The selected TTS model is not
   available for this provider; check the model in STTS Settings" instead of
   falling to the generic `generation_failed` / "Unexpected TTS generation
   failure; retry" bucket. Test:
   `Tests/TTS/test_tts_unknown_model_error_copy.py`.

3. **The stuck ⏹ was deterministic after all**: `handle_tts_complete_event`'s
   error branch reset legacy `ChatMessage` widgets only; the Console
   transcript renders its action row from the screen's
   `_console_speaking_message_id`, which nothing cleared on failure. The
   error branch now walks `screen_stack` for the screen whose marker matches
   the failed message, clears it, and resyncs — message-id-guarded so an
   unrelated screen's state is untouched. Test added to
   `Tests/TTS/test_console_speak_autoplay.py` (existing `_FakeApp` idiom).

Live verification (dead-port TTS endpoint, mock LLM, one 🔊 click): failure
toast shown, the action row returned to 🔊, and the Logs screen recorded
exactly TWO DISTINCT single rows (backend "Network request failed" +
handler "TTS generation failed") where each was previously doubled.

Adjacent observation, out of scope here: a connection failure surfaces the
ValueError-bucket copy "TTS is not configured; open STTS Settings" (the
backend wraps network errors in ValueError), which mislabels a reachability
problem as a configuration one — pre-existing, noted for a future pass.
<!-- SECTION:NOTES:END -->
