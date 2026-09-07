---
id: TASK-2362
title: 'Realtime: fake-server drift closure list from final review M9'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04'
updated_date: '2026-08-05 04:04'
labels:
  - realtime
  - test-quality
dependencies: []
priority: low
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add fake-server tests pinning (a) input_audio_buffer.committed -> on_turn_committed, (b) voice under session.audio.output present/absent, (c) transcription.model == whisper-1 (strengthen shared handshake predicate). Mutation-verify each against a temporary revert of the production line.
2. Extend Tests/LLM_Calls/openai_realtime_probe.py with an --audio mode that sends synthetic PCM and manually commits, fix its docstring to match reality (text mode default, audio mode opt-in), never print raw payloads. Run both modes live to confirm.
3. Move the singular input_token_details ground truth (and the newly observed audio/text usage split + transcription duration usage shape) from provider_usage.py's comment into openai_session.py's header, correcting the header's own over-claim about which events the original probe could reproduce.
4. Add a direct Tests/Chat/test_provider_usage.py test for the singular input_token_details alias (previously only covered via the UI wiring suite).
5. Tick ACs, add Implementation Notes, run full targeted suite + contract trio.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five drift items were resolved by ASSERTING them properly (none needed the
"intentionally uncovered" fallback):

- **(a) `input_audio_buffer.committed` -> `on_turn_committed`**: added
  `test_input_audio_buffer_committed_fires_on_turn_committed` to
  `Tests/LLM_Calls/test_openai_realtime_session.py`. Mutation-verified: temporarily
  replacing `_on_input_committed`'s `self._safe_invoke(...)` line with `pass` fails
  exactly this test (`assert 0 == 1`), confirmed then reverted (AC #2).
- **(b) `voice` under `session.audio.output` unasserted**: added
  `test_configured_voice_is_sent_under_audio_output` (asserts presence when
  configured) and `test_unset_voice_omits_the_voice_key` (asserts the key is
  absent, not merely falsy, when unconfigured -- mirrors the turn-detection
  tests' "omitted, not defaulted" discipline). Mutation-verified: deleting
  `_build_session_update`'s `output["voice"] = ...` line fails the first test.
- **(c) `whisper-1` only checked "enabled"**: strengthened the shared
  `_make_is_session_update` handshake predicate (used by ~30 of the suite's
  tests) to assert `transcription.model == "whisper-1"` as a literal, not by
  importing `openai_session._TRANSCRIPTION_MODEL` (which would make the
  assertion tautological -- it would silently "pass" even if production
  quietly switched models with nothing re-confirming the new one live).
  Mutation-verified: changing the production constant to a different string
  fails 32/33 tests in the module.
- **(d) probe script docstring/behavior mismatch**: rewrote
  `Tests/LLM_Calls/openai_realtime_probe.py`. The module and `_run_probe`
  docstrings claimed "audio+text modalities" (a shape the live GA endpoint
  rejects outright, per this same session's own header) while the code only
  ever sent a TEXT turn and no audio at all -- so it could never reproduce the
  `input_audio_buffer.*` observations `openai_session.py`'s header attributed
  to it. Fixed by making the claims true rather than merely restating them:
  the script gained a `--audio` mode (synthetic tone, manual
  `input_audio_buffer.commit` for probe determinism -- `turn_detection: null`,
  not production's `server_vad`/`semantic_vad`, documented as a deliberate
  probe-only simplification) alongside the original text-only default, and
  both docstrings now describe what each mode actually sends and can/cannot
  observe. Ran BOTH modes live three times each against the real GA endpoint
  to confirm; `--audio` reliably reproduced `input_audio_buffer.committed`,
  `conversation.item.input_audio_transcription.completed` (with its usage
  field), and `response.done`'s full usage shape. Never prints a raw event
  payload (only `type` plus specific safe fields), matching the existing
  `_safe_error_detail` discipline.
- **(e) singular `input_token_details` claim only in a provider_usage.py
  comment**: moved the ground truth into `openai_session.py`'s header (new
  "USAGE ground truth" section, dated and citing the three `--audio` probe
  runs above) -- which also let the header self-correct its own earlier
  over-claim that the ORIGINAL (pre-fix) probe script had observed
  `input_audio_buffer.*` events it structurally could not have sent. Added
  `test_realtime_singular_input_token_details_alias_maps_cached` directly to
  `Tests/Chat/test_provider_usage.py` (previously this branch's only
  mutation-covering test lived in `Tests/UI/test_console_realtime_wiring.py`,
  reachable only through the full Console wiring harness).

The same live probe runs also surfaced the two gaps TASK-2363 closes
(transcription-duration usage invisible to `on_usage`; audio/text token split
folded together) -- documented in the same new header section so 2363 did not
need its own separate live-probing pass.

Modified: `tldw_chatbook/LLM_Calls/realtime/openai_session.py` (header only,
no functional change), `Tests/LLM_Calls/test_openai_realtime_session.py`,
`Tests/LLM_Calls/openai_realtime_probe.py`, `Tests/Chat/test_provider_usage.py`.

Verification: `Tests/LLM_Calls/test_openai_realtime_session.py` (33 passed,
+4 vs. baseline), `Tests/Chat/test_provider_usage.py` (12 passed, +1 vs.
baseline -- the singular-alias test only; the audio/text-split fields it
also surfaced belong to and land in TASK-2363's own commit), contract trio
(`Tests/Chat/test_console_hands_free.py`, `Tests/UI/test_console_hands_free_
wiring.py`, `Tests/UI/test_console_dictation.py`) byte-identical and green
(136 passed).
<!-- SECTION:NOTES:END -->

## Description (the why)

The scripted fake WS server encodes the live-probed ground truth, but the V4 final review
(M9) listed residual drift: `input_audio_buffer.committed` is dispatched in production but
never emitted by any fake script (deleting the dispatch stays green); `voice` under
session.audio.output is sent but unasserted; `whisper-1` is documented live-confirmed but
the fake only checks transcription is enabled; the probe script's docstring still describes
a pre-GA session shape and sends no audio, so it cannot reproduce the input_audio_buffer
observations its header attributes to it; the singular `input_token_details` live-claim
lives only in provider_usage.py's comment.

## Acceptance Criteria (the what)

- [x] Each listed drift item is either asserted by the fake/probe or explicitly documented
      as intentionally uncovered.
- [x] Deleting the `input_audio_buffer.committed` dispatch fails a test.
- [x] The probe script's docstring matches what it actually sends.
