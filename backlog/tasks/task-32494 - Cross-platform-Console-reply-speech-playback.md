---
id: TASK-32494
title: Cross-platform Console reply speech playback
status: Done
assignee:
  - '@robert'
created_date: '2026-09-11 23:51'
updated_date: '2026-09-11 23:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console reply speech (hands-free loop and Speak replies) is silently dead on stock Linux: cloud TTS defaults to MP3, the built-in streaming sink only plays PCM/WAV, and the legacy fallback shells out to system player binaries (mpv/mplayer/ffplay/aplay/paplay) that stock Fedora does not ship. Playback failure after successful synthesis produces no user-visible signal (sequencer skips failed utterances; the one-toast latch only covers synthesis failures). Fix: format-aware player capability probing (incl. pw-play), adaptive response-format rewrite to WAV when the resolved format cannot be played locally, and loud, remedied failure surfacing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Format-aware player probe with pw-play in the chain and per-player format support, verified by unit tests (Tests/TTS/test_audio_player_formats.py: catalogue, pw-play/paplay conservative sets, capability-beats-order, play() refuses undecodable formats without spawning)
- [x] Adaptive rewrite: a Console speech request whose resolved format is not sink-eligible and not playable by any local player is re-issued as wav (provider permitting); macOS/mpv installs unchanged, verified by unit tests (Tests/TTS/test_playback_capability.py + Tests/TTS/test_speech_format_adaptation.py: hands-free adapts mp3→wav with sink, keeps mp3 with mpv, ad-hoc requests untouched)
- [x] Playback failure surfaces one user-visible toast per app run with a concrete remedy (install a player or switch Output format to WAV), verified by unit tests (test_speech_format_adaptation.py::TestPlaybackFailureRemedy: latch fires once across utterances, play() never runs, on_finished(False) keeps the loop moving)
- [x] Hands-free reply escalation: a reply where every utterance failed still produces a visible toast, verified by unit tests (Tests/UI/test_console_hands_free_wiring.py::test_reply_with_only_failed_utterances_escalates_once: loop still returns to listening, exactly one escalation notice)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read existing audio_player tests + preferences/effective-settings seams\n2. TDD: format-aware player probe (pw-play, per-player formats) in TTS/audio_player.py\n3. TDD: TTS/playback_capability.py helper (sink_available OR capable player per format)\n4. TDD: adaptive wav rewrite at Console speech request build when format unplayable\n5. TDD: playback-failure remedy toast (once per app run) + hands-free whole-reply escalation\n6. Targeted test runs; update task notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- **Approach.** TDD throughout (every unit watched RED first). Three layers: a format-aware Linux player catalogue in `TTS/audio_player.py` (`_LINUX_PLAYER_CATALOGUE`, `find_player_for_format`, `player_supported_formats`; `play()` now re-selects per artifact suffix and refuses formats the selected player cannot decode instead of spawning garbage); a new `TTS/playback_capability.py` combining `sink_available()` with the catalogue (`locally_playable_formats`, `format_playable_locally`, `adapt_console_speech_format`, `playback_remedy`); and wiring.
- **Adaptive rewrite** lives in `TTSEventHandler._generate_tts`: Console speech requests only (`on_finished` set = hands-free utterance, or `playback_lifecycle` set = automatic Speak replies) whose resolved preferences format is unplayable here are re-issued as WAV via a new `synthesize_default(response_format_override=...)` parameter that mirrors the existing `voice_override` seam (`TTS/request_admission.py` folds it into the explicit `TTSSelectionOverrides`, so it flows through the same per-provider validation — an unsupported wav fails loudly through normal error paths). Ad-hoc/explicit/character-profile requests are deliberately untouched; profile-driven formats get the remedy toast instead of a silent rewrite. macOS (afplay) and mpv installs are byte-identical in behavior.
- **Failure surfacing.** `_play_utterance_legacy_artifact` pre-checks playability: when the artifact's format provably cannot play, it skips the spawn, posts one remedy toast per app run (`_playback_remedy_notified` latch, mirroring the dictation override notices), cleans the artifact up, and reports `on_finished(False)` so the loop keeps moving. The hands-free wiring additionally tracks `utterances_dispatched`/`utterances_failed` per reply and escalates once when a COMPLETED reply's every utterance failed (`_maybe_escalate_all_failed_reply`, evaluated from both settle points — completion tap and sequencer-drain tap — because either can land first; gated on the FSM reaching `listening` so the counters are final).
- **Conservative format sets** are deliberate: `paplay`/`pw-play` decode via libsndfile whose MP3 support is version/build-flag dependent (libsndfile ≥1.1.0, 2022), so they are only selected for wav/flac; false negatives are safe (a better format is requested instead), false positives would resurrect the silent-failure bug. `aplay` is wav-only.
- **Files.** `tldw_chatbook/TTS/audio_player.py`, `tldw_chatbook/TTS/playback_capability.py` (new), `tldw_chatbook/TTS/request_admission.py`, `tldw_chatbook/TTS/TTS_Generation.py`, `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`, `tldw_chatbook/UI/Console_Modules/hands_free.py`; tests: `Tests/TTS/test_audio_player_formats.py` (new), `Tests/TTS/test_playback_capability.py` (new), `Tests/TTS/test_speech_format_adaptation.py` (new), `Tests/UI/test_console_hands_free_wiring.py` (extended); `Tests/TTS/test_console_audio_cpp_native.py` fake updated for the new `synthesize_default` kwarg.
- **ADR required: no** — bug fix plus additive seams (one optional request parameter, one new leaf module); no schema, no boundary restructuring. Documented here instead.
- **Verification.** 103 passed across the playback/wiring/sequencer surface; full `Tests/TTS` green (1176 passed, 2 skipped) after the fake-service signature update.
