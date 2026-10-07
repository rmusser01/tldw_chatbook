# Speech: TTS, voice profiles, dictation, and hands-free

This document describes the speech stack: the TTS service and its lease-based adapter registry, the layered voice selection model, the encrypted-ish profile store, local and remote backends, PCM streaming vs file playback, STT/dictation into the Console composer, and the hands-free loop.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| TTS service | `TTS/TTS_Generation.py` | `TTSService` — async generation coordinator; admission, settings publication, audio.cpp lifecycle; max 4 concurrent operations |
| Adapter registry | `TTS/adapter_registry.py` | `TTSAdapterRegistry`, `TTSAdapterLease`, `TTSReconfigurationTicket`, `ReconfigureResult` (UNCHANGED/CHANGED/SUPERSEDED), `stage_provider_configuration`, `acquire(expected_revision=…)` |
| Composition | `TTS/adapter_bootstrap.py` | `build_default_tts_service(app_config)` — app-scoped `AudioCppSupervisor`, registry, profile store wiring |
| Native adapter | `TTS/adapters/audio_cpp.py`, `audio_cpp_supervisor.py`, `audio_cpp_config.py` | `AudioCppAdapter` (external or managed audio.cpp server), `AudioCppSupervisor` (at most one lazily launched child process), `AudioCppConfig` |
| Layered selection | `TTS/effective_settings.py` | `TTSEffectiveSelectionSnapshot`; non-studio layer order **explicit > character_profile > default_profile > global** (+ provider fallback); studio requests reject non-studio layers |
| Global prefs | `TTS/preferences.py` | `TTSPreferencesSnapshot.from_settings` reads `[app_tts]`; audio.cpp constrains format to wav, speed 1.0 |
| Profiles | `TTS/profile_repository.py`, `profile_schema.py`, `profile_service.py`, `profile_types.py` | `TTSProfileRepository` (one serialized connection), `TTSGenerationProfile` (revision-CAS), `TTSProfileVerificationEvidence` |
| Backends | `TTS/backends/` | openai, elevenlabs, alltalk, kokoro (ONNX/PyTorch), chatterbox (+ isolated subprocess variant), higgs |
| Playback | `Audio/streaming_sink.py`, `TTS/pcm_stream.py`, `TTS/audio_player.py`, `TTS/playback_capability.py` | `StreamingPcmSink`, `sink_plan()`, `SimpleAudioPlayer`/`AsyncAudioPlayer`, `locally_playable_formats()` |
| STT catalogue | `Utils/local_stt_providers.py` | `LOCAL_PROVIDER_MODULES` — the single source of truth for local STT providers |
| Dictation | `Audio/dictation_service.py`, `Chat/console_voice_input.py`, `UI/Console_Modules/dictation.py` | `DictationService` FSM, headless `ConsoleVoiceInputController` (probe/resolve/classify), `ConsoleDictationController` |
| TTS event handler | `Event_Handlers/TTS_Events/tts_events.py` | `TTSEventHandler`, `TTSMessageSpeechRequestEvent`, `TTSPlaybackEvent`, `TTSGlobalOverrideDecisionEvent`, `ConsoleTTSDestination` |
| Speech snapshots | `Chat/console_speech.py`, `Chat/console_chat_store.py` | `TTSMessageSpeechSnapshot`, `issue_tts_message_speech_snapshot` / `validate_tts_message_speech_snapshot` |
| Hands-free | `Chat/console_hands_free.py`, `UI/Console_Modules/hands_free.py`, `Chat/reply_sentence_sequencer.py`, `Chat/console_auto_speak.py` | headless FSM (`idle|listening|countdown|awaiting_reply|speaking`), `ConsoleHandsFreeController`, `SentenceSequencer`, `decide_auto_speak()` |
| Settings | `UI/Screens/settings_speech_tts.py`, `UI/Speech/` | the speech-tts settings category + Speech Lab |

## Voice selection and provider reconfiguration

Selection precedence (non-studio): explicit > character assignment > default profile > global `[app_tts]` > provider fallback. Character speech resolves through `CharacterTTSRequestResolver`; only when resolution lands on `"global"` is the app default profile consulted.

The adapter registry is **lease-based with a revision + generation model**: every synthesis admits with `expected_revision`; a provider reconfiguration stages a config under a registry-wide generation (SUPERSEDED if a newer generation already exists), drains leases, runs the lifecycle action, and promotes the staged config. Failed exclusive transitions seal the provider unavailable (`TTSProviderUnavailableError` at acquire). Verification evidence additionally pins `provider_configuration_revision`, so a provider reconfiguration invalidates previous profile verification.

Profiles live in a dedicated SQLite store (`config.get_tts_profiles_db_path()`, default `<user_data>/tldw_chatbook_tts_profiles.db`): UUID ids, NFKC-casefold name uniqueness, monotonic revisions with CAS updates, v2/v3 migrations with backups, portable bundles with hostile-archive admission (exact member order, size caps, expansion-ratio limit), and private clone-reference assets (ADR-051). The store requires the `SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE` capability and fails closed without it.

## Backends

| Backend | Transport | Notes |
| --- | --- | --- |
| audio.cpp (native adapter) | HTTP; external URL or managed child process | managed mode launches/supervises one process; transient HTTP statuses retried once; PCM16 WAV bodies validated |
| openai | OpenAI-compatible `POST …/audio/speech` | custom endpoints treated as keyless; key fallback chain env → config → `[api_settings.openai]` → legacy sections |
| elevenlabs | streaming endpoint | default voice/model/format with stability/similarity knobs |
| alltalk | local server, OpenAI-compatible | legacy default voice remapped; httpx logging hard-filtered |
| kokoro | in-process ONNX or PyTorch | atomic model downloads; Windows onnxruntime pre-check |
| chatterbox | local; via isolated subprocess | subprocess speaks JSON+base64 over pipes; stdout of the child is silenced before imports |
| higgs | in-process (`[HiggsSettings]`) | zero-shot cloning, multi-speaker, background music |

## Playback model

Two paths, never both for one response: **live PCM streaming** through `StreamingPcmSink` (PortAudio, prebuffered, `stop()` must reach silence within two audio blocks, no PortAudio calls from the callback), or **artifact file** played by an external player (`afplay`/mpv/…; format catalogues are deliberately conservative — `afplay` excludes opus/ogg because it silently "succeeds" without decoding). PCM is sink-only (no external player decodes bare PCM); `locally_playable_formats()` is re-probed at every call so mid-run player installs take effect. Legacy playback completion is a poll with a text-length-derived timeout because `AudioPlayerInfo.duration` is never populated.

## Speak action (dataflow)

1. Speak on an assistant message → the store **issues a snapshot** (`issue_tts_message_speech_snapshot`) — rejects non-assistant/incomplete/blank/off-active-path messages with bounded rejection codes ("Message changed before speech started; select Speak again.").
2. Prior speech is stopped via `TTSPlaybackEvent(action="stop")` with a settle outcome; a failed stop aborts the new request.
3. A `TTSPlaybackLifecycle` validator (generation, session id/epoch, snapshot revalidation) rides the `TTSMessageSpeechRequestEvent`.
4. The handler revalidates, prepares text, resolves voice authority, checks the auto-speak destination fingerprint (a changed destination fails with a confirm-again copy), admits (per-message cooldown; supersedes the current generation owner), and synthesizes through the registry lease.
5. Eligible responses stream via the sink; otherwise the artifact file plays externally. Completion events are discarded if their lifecycle is no longer current.

Auto-speak is fail-closed: unknown session, hands-free active, or a missing/differing destination fingerprint all suppress speech; per-conversation consent is stored as session metadata (destination SHA-256 fingerprint, consent version).

## Dictation (dataflow)

1. The Console mic button starts a streaming dictation session (16 kHz mono PCM, bounded 60 s / 1.92 MiB).
2. Provider resolution is **fail-closed**: the configured local provider must be installed (`[transcription] default_provider`, legacy `[STT_settings]` fallback); the `[dictation] model` override wins; non-faster-whisper providers get `model=None` so they never inherit a cross-provider model name (a real past 404 failure).
3. Partials stream into the composer (`set_voice_partial`); spoken commands with the `[dictation] command_prefix` (default "computer") can send, insert new-lines, or stop.
4. Stop (button, command, or limit) → `stop_and_transcribe()` blocks for the final transcript; failure offers a faster-whisper retry.
5. Insertion is caret-aware and trims only spaces/tabs (a full `.strip()` used to eat spoken `new-line` commands); for non-live sessions the draft is appended to the store and banked undo history is deliberately dropped.

Note: the `Event_Handlers/Audio_Events` dictation message family serves the legacy dictation window/voice widget — the Console mic path is `Chat/console_voice_input.py` + `UI/Console_Modules/dictation.py`.

## Hands-free loop

Engine fork at entry: realtime (opt-in `[realtime] enabled`) or the pipeline FSM; a forced realtime engine with realtime disabled is refused loudly, never downgraded; the fork never re-resolves mid-loop. The pipeline cycle: listen → VAD/silence or countdown ends capture → send → stream reply → `SentenceSequencer` splits confirmed sentences → per-utterance `speak_utterance` (cooldown-free, one failure toast per reply) → mic reopens. A 30 s awaiting-reply watchdog recovers if a send silently refused; Esc exits unconditionally. There is **no acoustic echo cancellation**: re-entry silences in-flight reply audio because the recognizer would transcribe the assistant's own voice; in the degraded no-VAD mode, voice-cancel/barge-in are silently inert while keypress/spoken stop still work.

## Config keys (exact sections)

`[app_tts]` (the TTS section — there is **no** `[tts]`/`[speech]`), `[HiggsSettings]`, `[dictation]` (model, command_prefix, handsfree_engine, privacy.local_only), `[transcription]` (default_provider/model/language, provider knobs; legacy `[STT_settings]`), `[diarization]`, opt-in `[realtime]`, `[database] tts_profiles_db_path`. Auto-speak preferences are per-conversation session metadata, not config.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| No playable format locally | Capability gate refuses synthesis; hands-free warns once with the remedy copy |
| Sink failure mid-stream (WAV) | Silent fallback to the already-written file (PCM responses surface the error instead) |
| Provider reconfiguration during playback | Transition waits for the lease (bounded by one utterance's playback) |
| Stale speech snapshot | Bounded rejection code; no content leaks into the rejection |
| Dictation provider uninstalled | Resolution returns unavailable with reason/remedy; never falls back silently |
| Profile store integrity failure | Coded repository errors; migrations back up first; capability gate fails closed |
| httpx logging | Hard-filtered in network adapters to keep URLs/keys out of logs |

## Governing decisions

ADR-023 (TTS adapter registry + audio.cpp runtime boundary), ADR-028 (character TTS profile ownership), ADR-039 (global and studio settings ownership), ADR-040 (Speech Lab current result/auto-play), ADR-050 (audio.cpp generated model setup), ADR-051 (private TTS clone references), ADR-012 (provider credentials), ADR-025 (shared STT artifacts). Note duplicate ADR numbers exist — cite by filename. User guides: `Docs/User_Guide/console/voice-and-hands-free.md`, `Docs/User_Guide/console/attachments-images-voice.md`, `Docs/Features/Speech-Services-Guide.md`.

## Verified gotchas

1. `TTS/__init__.py` eagerly imports Textual transitively — zero-Textual modules (`Audio/streaming_sink.py`) must never import `TTS.pcm_stream` at module scope.
2. The local-STT provider list once existed twice and drifted (wrong model warmed, another model transcribed) — `Utils/local_stt_providers.py` is now the single source.
3. Streaming holds the provider lease through playback; reconfiguration can be delayed by one utterance.
4. Chatterbox's child process silences stdout/stderr before imports and speaks JSON on the dup'ed fd — any library print corrupts the protocol.
5. Voice bundles are parsed with manual ZIP structs (member order, caps, 100× expansion ratio) — hostile-archive admission, not `zipfile` defaults.
