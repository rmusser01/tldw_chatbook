---
id: TASK-2363
title: 'Realtime: usage completeness — transcription duration and audio-token split'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04'
updated_date: '2026-08-05 04:21'
labels:
  - realtime
  - cost
dependencies: []
priority: medium
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend ProviderUsage with audio_input/audio_output (int, subset of the existing uncached_input/output totals -- additive, not summed separately into total_tokens) and transcription_seconds (float, a distinct unbillable-by-default unit). Parse input_token_details/output_token_details audio_tokens in from_provider_payload's existing Realtime/Responses branch. Update to_json/from_json/plus() for the new fields.
2. Add RealtimeCallbacks.on_transcription_usage; fire it from _on_input_transcript_completed when the event carries a usage field (raw passthrough, no interpretation at the session layer -- matches on_usage's existing division of labor). Fake-server tests for fires-with-usage / does-not-fire-without.
3. Wire chat_screen.py: register the callback, implement _on_console_realtime_transcription_usage attaching ProviderUsage(transcription_seconds=...) to session.user_row_id (not the assistant's last_reply_row_id -- this is about the user's spoken input, not the assistant's reply). Guard against overwriting an already-set usage on a row that moved to a new turn.
4. UI wiring suite: add FakeRealtimeSession.fire_transcription_usage; test duration attaches to the user row; test response.done's audio/text split attaches distinctly to the assistant row's usage (real captured GA payload as fixture).
5. Tick ACs (including explicitly filing/declining a cost-chip follow-up per AC #3), Implementation Notes, run targeted suite + contract trio.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach**: extended `ProviderUsage` additively (three new defaulted fields:
`audio_input`, `audio_output`, `transcription_seconds`) rather than inventing a
parallel metadata channel, since it is already the codebase's one normalized
"turn usage" abstraction, already flows through `to_json`/`from_json`/`plus()`/
`set_message_usage`, and `pricing_catalog.py` reads named fields explicitly
(never `asdict()`-iterates), so the new fields are automatically inert for
billing without any extra work -- satisfying "explicitly-unbillable metadata"
by construction, not by a separate guard.

- **Transcription duration (gap 1, T2-F12)**: added `RealtimeCallbacks.
  on_transcription_usage` (additive; `on_usage` untouched) and fired it from
  `openai_session.py`'s `_on_input_transcript_completed` when the event's
  `usage` field is present -- raw passthrough, no interpretation at the
  session layer, matching `on_usage`'s own division of labor. NOT routed
  through the existing `on_usage`/`_on_console_realtime_usage` path: that
  handler targets `last_reply_row_id` (the ASSISTANT's row) and
  `ProviderUsage.from_provider_payload` doesn't recognize a
  `{"type": "duration", ...}` shape at all (would silently return None) --
  either bug would have misattributed or dropped it. New wiring handler
  `_on_console_realtime_transcription_usage` targets `session.user_row_id`
  instead (this is about the user's spoken input), with the same "never
  overwrite an already-set usage" guard `_on_console_realtime_input_
  transcript` already uses for late-arriving transcript text, extended to
  usage.
- **Audio/text token split (gap 2, F9's other half)**: `ProviderUsage.
  from_provider_payload`'s existing `input_token_details`/`input_tokens_
  details` branch (shared by Realtime and the Responses API) now also reads
  `audio_tokens` off both `input_token_details` and the new `output_token_
  details`, live-confirmed via three `--audio` probe runs (task-2362) to
  split BOTH input and output into `text_tokens`/`audio_tokens`.
- **Cost-chip integration (AC #3)**: NOT wired -- filed as TASK-2390
  ("Realtime: cost-chip integration for audio-token and transcription-
  duration usage"), per this task's own scope note that cost-chip wiring is
  explicitly out of scope. `pricing_catalog.py`'s per-mtok model cannot
  represent per-audio-minute billing as-is, which is real follow-up work,
  not a trivial wire-through.

**Modified**: `tldw_chatbook/Chat/provider_usage.py`, `tldw_chatbook/LLM_Calls/
realtime/protocol.py`, `tldw_chatbook/LLM_Calls/realtime/openai_session.py`,
`tldw_chatbook/UI/Screens/chat_screen.py`; tests in `Tests/Chat/
test_provider_usage.py`, `Tests/LLM_Calls/test_openai_realtime_session.py`,
`Tests/LLM_Calls/test_realtime_protocol.py`, `Tests/UI/
test_console_realtime_wiring.py`.

**Verification**: TDD RED-first throughout (confirmed every new assertion
failed against pre-change code before implementing). Targeted suite: `Tests/
Chat/test_provider_usage.py` (14 passed), `Tests/LLM_Calls/
test_openai_realtime_session.py` (35 passed), `Tests/LLM_Calls/
test_realtime_protocol.py` (15 passed), `Tests/UI/
test_console_realtime_wiring.py` (58 passed), `Tests/Chat/
test_console_realtime_loop.py` (unaffected, pure FSM). Cross-cutting
`ProviderUsage` consumers checked for regressions: `Tests/Chat/
test_console_chat_store.py`, `test_console_cost_tracker.py`, `Tests/
LLM_Calls/test_pricing_catalog.py`, `Tests/Chat/test_console_variant_
stream.py`, `Tests/UI/test_console_resume_active_path.py`, `Tests/Chat/
test_console_provider_gateway.py` -- all green, no field-count assumptions
broken (equality tests compare full dataclasses, but every existing
construction omits the new fields, so both sides default to 0/0.0
identically). Contract trio (`Tests/Chat/test_console_hands_free.py`, `Tests/
UI/test_console_hands_free_wiring.py`, `Tests/UI/test_console_dictation.py`)
byte-identical, 103 passed. Broader sweep `Tests/Chat/ Tests/LLM_Calls/`
(3474 passed, 66 skipped): 1 failure + 6 errors, all in `test_chat_functions.
py::TestChatApiCall` and `test_console_generation_actions.py`, neither file
touched by this change -- reproduced identically in complete isolation
(Textual internal `_nodes` API drift and a missing fixture), confirmed
pre-existing and unrelated.
<!-- SECTION:NOTES:END -->

## Description (the why)

Input-audio transcription events carry `usage: {type: duration, seconds: N}` that never
reaches `on_usage` (T2-F12), and realtime `response.done` usage folds audio tokens into
text counts in `ProviderUsage` (final review F9 fixed the cached-token half only). Realtime
is billed per audio minute; the Console cost chip cannot be honest about it until these are
captured distinctly.

## Acceptance Criteria (the what)

- [x] Transcription duration usage is captured onto the turn (or explicitly recorded as
      unbillable metadata) rather than dropped.
- [x] Audio vs text token counts from realtime responses are recorded distinctly.
- [x] Cost-chip integration for realtime sessions is either wired or filed as its own task
      with the captured fields it needs.
