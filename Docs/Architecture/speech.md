# Speech: TTS, voice profiles, dictation, and hands-free

This document describes the speech stack: the TTS service and its lease-based adapter registry, the layered voice selection model, the private SQLite profile store, local and remote backends, PCM streaming vs file playback, STT/dictation into the Console composer, and the hands-free loop.

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

## Audiobook chapter source authority

`UI/STTS_Window.py::AudioBookGenerationWidget` keeps the complete file, Notes, or Conversation import in its editable TextArea and working source. Programmatic import echoes therefore retain the full source for chapter detection and Generate, including text beyond character 1000; deliberate edits replace that working source and clearing it blocks Generate (TASK-36130). Chapter publication generation advances when the current TextArea source changes, including clear and the debounce interval. Dispatch also advances it, so a slow superseded worker cannot replace current chapters or announce stale success. A same-source programmatic import echo keeps its dispatch valid (TASK-36100).

Pending file reads additionally retain the current editor identity, live text and
working-source snapshot across the off-loop read. Completion publishes only if
the owning widget remains attached and all three still match. This preserves
accepted edits or clear even while the parent's Changed message is queued, and
a discarded import emits no success notification (TASK-36363).


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

Two paths, never concurrently for one response: **PCM output** through `StreamingPcmSink` (raw streaming or an eligible complete PCM16 WAV body; PortAudio, prebuffered, `stop()` must reach silence within two audio blocks, no PortAudio stream operations from the callback), or an **artifact file** played by an external player (`afplay`/mpv/…; format catalogues are deliberately conservative — `afplay` excludes opus/ogg because it silently "succeeds" without decoding). PCM is sink-only (no external player decodes bare PCM); `locally_playable_formats()` is re-probed at every call so mid-run player installs take effect. Legacy playback completion is a poll with a text-length-derived timeout because `AudioPlayerInfo.duration` is never populated.

## Pure speech presentation contracts

`Audio/speech_presentation.py` supplies stdlib-only PCM16 energy windows, a raw
silence gate with finite dBFS smoothing, immutable owner/snapshot values and a
thread-safe latest-value channel. PCM windows follow accepted source frames,
including split channel samples and fractional frames per 20 ms window; this
module does not open a device, run a thread or assign an output clock.

`SpeechPresentation` belongs to one exact host-issued owner instance and cannot
be retargeted. It rejects old generations/observations, preserves confirmed pause
and terminal transport, and closes stale (>100 ms), future or invalid progress.
Qualified silence has a closed mouth and no finite level while retaining valid
timing; unknown timing is separate. Clock failure degrades observation without
raising into audio callers. No waveform/history is stored in the channel.

`StreamingPcmSink(..., presentation=...)` prepares accepted PCM on its existing
notification thread and publishes every nominal 20 ms. Source consumption and
inserted device silence remain separate, including half-sample chunk boundaries
and partial zero-padded blocks. Validated backend current/DAC timestamps map to
one monotonic output timeline; poses are selected only inside their actual output
interval. The device-tail horizon is at most one second. Missing, nonfinite,
backward, excessive or hardware-underflow timing selects closed/status-only
observation, with audio unchanged. Queued preparation is bounded by the existing
queued-audio capacity plus that tail; readers hold only one snapshot. Underrun
notifications coalesce behind a blocked listener. A stalled publisher relies on
snapshot freshness rather than accumulating UI events.

Natural drain requests sounddevice `CallbackStop` only through the thin backend
wrapper. The sink stays draining, registered and abortable until `finished_callback`
confirms pending audio played; then it closes without aborting, publishes drain
and releases ownership. Immediate Stop/failure still aborts; late acknowledgements
cannot change the terminal result. The existing buffered-duration plus five-second
pump watchdog bounds missing acknowledgement. `settle()` settles queued events,
not a pending device tail. A backend finishing during open cannot evict a healthy
previous sink. Complete eligible WAV routing is reused, including existing scoped
cleanup and fallback; no second player is introduced.

Sink output epochs are distinct from request/session generations. Local sink
termination retires its output epoch before clearing its level, so an in-flight
pose with an equal timestamp cannot reopen the mouth. Duplicate old cleanup
cannot invalidate a newer output epoch. Termination invalidates its level immediately; the request adapter projects the
logical terminal state after it knows whether a WAV failure falls back to legacy
playback. This preserves one owner through fallback without reopening a terminal
channel. Console request adapters/rendering remain subsequent implementation units;
these optional sink contracts alone do not enable a moving Console portrait.

## Speak action (dataflow)

1. Speak on an assistant message → the store **issues a snapshot** (`issue_tts_message_speech_snapshot`) — rejects non-assistant/incomplete/blank/off-active-path messages with bounded rejection codes ("Message changed before speech started; select Speak again.").
2. Prior speech is stopped via `TTSPlaybackEvent(action="stop")` with a settle outcome; a failed stop aborts the new request.
3. A `TTSPlaybackLifecycle` validator (generation, session id/epoch, snapshot revalidation) rides the `TTSMessageSpeechRequestEvent`.
4. The handler revalidates, prepares text, resolves voice authority, checks the auto-speak destination fingerprint (a changed destination fails with a confirm-again copy), admits (per-message cooldown; supersedes the current generation owner), and synthesizes through the registry lease.
5. Eligible responses stream via the sink; otherwise the artifact file plays externally. Completion events are discarded if their lifecycle is no longer current.

Auto-speak is fail-closed: unknown session, hands-free active, or a missing/differing destination fingerprint all suppress speech; per-conversation consent is stored as session metadata (destination SHA-256 fingerprint, consent version).

## Foreground Speech Recognition capture

The Speech Lab's `ImprovedDictationWindow` owns its lazy dictation service.
Its idempotent `cleanup()` revokes transcript, state, error and command callbacks
before invoking the existing nonjoining `abandon()` for active or failed-start
processing. STTS view replacement invokes this cleanup seam; direct removal and
shutdown also run it before `on_unmount()` awaits pending settings writes.
Processor callbacks use the widget message pump to publish on the UI thread and
recheck retirement there, so an already-entered callback also loses publication
authority. Saved callbacks cannot update retired controls or a replacement view.
Start and completed Stop advance capture authority so queued updates cannot cross
into a restarted capture. Stop renders its complete returned transcript after the
service clears private buffers. A paused capture resumes into the same normal
Stop path so it returns its transcript and releases the recorder. Later cleanup
does not stop the recorder again. An unopened view never constructs a service
(TASK-36693; ADR-025/033).

The Dictation action buttons support focused Space and inherited Enter. The
child consumes each recognized `Button.Pressed` after handling it so STTS cannot
manually forward the same bubbled event back to the child. Unrecognized and
container-owned controls retain their normal route (TASK-36694; ADR-025/031).

## Dictation (dataflow)

1. The Console mic button starts a streaming dictation session (16 kHz mono PCM, bounded 60 s / 1.92 MiB).
2. Provider resolution (`Chat/console_voice_input.py::resolve()`) returns unavailable only when no local provider is installed. It uses the configured provider (`[transcription] default_provider`, legacy `[STT_settings]` fallback) when installed; otherwise it selects the first installed provider in `LOCAL_PROVIDER_MODULES` declaration order. Replacing a configured provider emits `VoiceProviderOverridden`, which the Console reports once per app run with a warning toast naming both providers. At startup, `app.py` also records a WARNING `speech_stack_available` diagnostic naming both providers: status `fallback` when VAD is available, or `degraded-fallback` when VAD is missing. The `[dictation] model` override wins; non-faster-whisper providers get `model=None` so they never inherit a cross-provider model name (a real past 404 failure).
3. Partials stream into the composer (`set_voice_partial`); spoken commands with the `[dictation] command_prefix` (default "computer") can send, insert new-lines, or stop.
4. Stop (button, command, or limit) → `stop_and_transcribe()` blocks for the final transcript; failure offers a faster-whisper retry.
5. Insertion is caret-aware and trims only spaces/tabs (a full `.strip()` used to eat spoken `new-line` commands); for non-live sessions the draft is appended to the store and banked undo history is deliberately dropped.

Note: the `Event_Handlers/Audio_Events` dictation message family (`DictationEvent` + `VoiceCommandEvent`) serves the legacy dictation window (the voice widget was removed, task-34898) — the Console mic path is `Chat/console_voice_input.py` + `UI/Console_Modules/dictation.py`.

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
| Configured dictation provider uninstalled | Uses the first installed local provider in declaration order, with the substitution toast and startup warning naming both providers; no installed local provider means unavailable with reason/remedy |
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
