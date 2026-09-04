# Meeting transcription — design

- **Date:** 2026-09-04
- **Status:** Approved in brainstorming; awaiting spec review
- **Scope:** Phase 1 of a meeting-transcription feature for tldw_chatbook: record a
  Zoom call or an in-person meeting from inside the TUI, show a live transcript
  labelled "You" / "Others" during a call, and land the result in the Library as a
  normal audio media item with a post-meeting diarized transcript.
- **Out of scope for phase 1:** live per-speaker diarization (phase 2, MOSS
  candidate), pushing to tldw_server's meetings feature (server sink, follow-on),
  Windows loopback verification (own task), importing Zoom's own recordings.

## 1. Decisions taken during brainstorming

| Question | Decision |
|---|---|
| What is the deliverable | Local recording → Library media item now; transport designed so the same capture can push to the server's meetings session later |
| How to hear the remote Zoom participants | Native system-audio capture (no virtual-device requirement), with a user-chosen input device as the fallback |
| What the user sees during the meeting | Live rolling transcript labelled You/Others in a call; live per-speaker labels are phase 2 |
| Platforms in v1 | macOS and Linux verified here; Windows loopback ships behind a verify-on-Windows task |
| Where it lives | Its own Meetings screen and tab route, not Console or Library |
| Transcription topology | One dictation pipeline over a mixed stream, attribution from per-source energy. Two parallel pipelines were rejected because the shared local-STT executor reserves one dictation capture at a time (`STT/dispatch_coordinator.py`, "Another dictation capture is already reserved") |

## 2. Existing building blocks reused

- `Audio/recording_service.py` — `AudioRecordingService`: PyAudio-primary /
  sounddevice-fallback mic capture, device enumeration, WebRTC VAD with 240 ms
  pre-roll. Reused for the mic and for the virtual-device fallback.
- `Audio/dictation_service_lazy.py` — `LazyLiveDictationService`: silence-gated
  segmenting, streaming (parakeet-mlx) / deferred (executor) / per-segment
  regimes, partial + final callbacks. Reused unchanged except for one kwarg.
- `Audio/realtime_mic_tap.py` — `RealtimeMicTap`: the on-frames contract and the
  `recorder_factory` injection pattern the new tap and capture copy.
- `Local_Ingestion/transcription_service.py` — `TranscriptionService`
  facade; constructed with `local_stt_dispatcher=None` it runs every provider
  in-process with no 60-second dictation ceiling.
- `Chat/console_voice_input.py` — `probe()` / `resolve()`: effective STT
  provider/model resolution honouring privacy local-only mode. Reused for the
  provider readout and the session's provider choice.
- `Library/library_ingest_jobs.py` — `LibraryIngestJobRegistry.submit(...)`:
  the post-meeting handoff. UI-thread-only by contract.
- `Local_Ingestion/audio_processing.py` — the offline audio pipeline with
  diarization, reached through the ingest job; local files are left in place.
- `TTS/audio_cpp_supervisor.py` — subprocess supervision pattern (spawn, ready
  line, diagnostics ring, shutdown deadline) copied for the capture helper.
- `Utils/path_validation.py` — recordings directory validation.
- Screen conventions: `UI/Navigation/base_app_screen.py` (`BaseAppScreen`),
  `UI/Navigation/screen_registry.py` (`ScreenRoute`), `Constants.py` tab ids,
  `ScreenStateStore` save/restore, app-owned lifecycles drained in
  `app._shutdown_app_owned_lifecycles`.

## 3. Components

New modules, all free of Textual imports except the screen:

### 3.1 `Audio/system_audio_tap.py` — `SystemAudioTap`

Delivers 20 ms PCM16 mono 16 kHz frames of system audio to an `on_frames`
callback, same contract as `RealtimeMicTap`.

- `probe() -> TapMode` runs before a meeting and returns one of
  `native_macos`, `native_parec`, `native_wasapi`, `virtual_device`, `unavailable`,
  plus a human reason string.
- **macOS 14.2+:** spawns the Swift helper (§3.6) and reads its stdout.
- **Linux:** spawns `parec --device=<default_sink>.monitor --format=s16le
  --rate=16000 --channels=1 --latency-msec=20`, falling back to `pw-record
  --target <default_sink>.monitor --rate 16000 --channels 1 --format s16 -`.
  The default sink comes from `pactl get-default-sink`. PortAudio on distro
  builds has no PulseAudio host API, so monitor sources never appear in the
  sounddevice device list; the subprocess route is the only reliable one.
- **Windows:** opens a sounddevice `InputStream` on the WASAPI device whose name
  ends in `[Loopback]` and matches the default output device, with
  `WasapiSettings(auto_convert=True)`. Lands behind the verify-on-Windows task;
  the resolver is a pure function over a device list and is unit-tested now.
- **Fallback (any OS):** a second `AudioRecordingService(retain_audio=False)`
  on the user-chosen input device (BlackHole, VB-Cable, etc.).
- Subprocess modes share one reader thread: reads stdout in 640-byte frames into
  a `queue.Queue(maxsize=50)` (1 s), dropping the oldest on overflow, and
  forwards to `on_frames`. A helper that exits mid-meeting is restarted once
  after 2 s; a second death puts the tap in `lost` state and it zero-fills.

### 3.2 `Audio/meeting_capture.py` — `MeetingCapture`

Duck-types the recorder surface `LazyLiveDictationService` uses, so the
service needs no knowledge of meetings:
`start_recording(callback) -> bool`, `stop_recording() -> None`,
`get_audio_level() -> float`, `get_audio_devices()`, `set_device(id) -> bool`,
`is_available() -> bool`, attributes `sample_rate = 16000`, `channels = 1`.

Owns the mic `AudioRecordingService(use_vad=False, retain_audio=False)` and,
in call mode, a `SystemAudioTap`. **The mic callback is the clock:** every mic
frame pulls one system frame from the tap queue (zeros if empty; backlog older
than 200 ms is discarded so the tracks stay within 200 ms of each other). No
mixer thread, no wall-clock tick. Per mic frame it:

1. writes the mic frame, the system frame and the mix to their `.pcm` files;
2. stores per-source RMS into the energy ring (100 ms buckets, 10 minutes);
3. advances `audio_position_s = bytes_written / (16000 * 2)`;
4. runs WebRTC VAD on the mix with the recorder's 240 ms pre-roll logic, opens a
   *speech run* on the first speech frame and closes it after
   `dictation.silence_threshold_seconds` without speech (the same setting the
   dictation service's silence gate reads, so runs and segments align);
5. hands speech frames to the dictation callback.

Mix = `clip(int32(mic) + int32(sys), -32768, 32767)` as int16 (numpy).
Room mode (no system source) writes only `mixed.pcm` and produces no labels.
Pause stops steps 1–5 but keeps devices open and keeps draining the tap queue
so nothing backlogs.

Meeting-specific surface: `levels() -> (mic_rms, sys_rms)`, `audio_position_s`,
`pause()`, `resume()`, `take_closed_runs() -> list[SpeechRun]`,
`dominant_source(start_s, end_s) -> "you" | "others" | "both"`.

### 3.3 `Audio/meeting_session.py` — `MeetingSession`, sinks, diarizer seam

`MeetingSession(capture_factory, dictation_factory, sinks, folder)`:

- `start(mode)` builds one `LazyLiveDictationService(recorder_factory=capture,
  transcription_service_factory=lambda: TranscriptionService(local_stt_dispatcher=None),
  enable_commands=False, ...)`, forces `privacy_settings["auto_clear_buffer"] = True`
  on it, and starts it with every callback except `on_command`.
- `pause()` / `resume()` forward to the capture.
- `stop() -> MeetingResult` stops the service (tail drain under its 30 s join
  budget), stops the capture, wraps `.pcm` into `.wav`, writes `meeting.json`,
  calls `on_stopped` on every sink.
- `segments: list[MeetingSegment]`, `subscribe(listener)`, `unsubscribe(...)`,
  `state`.

Segment building: a final consumes the oldest unconsumed closed speech run
(FIFO). If none is closed (streaming finals, or the service's 30 s hard segment
cap split a run) the window is `[run_start, capture.audio_position_s]`. Label =
`capture.dominant_source(start, end)` in call mode, `None` in room mode.
Partials get a live label from the last second. Each segment stores
`t_audio_start/end` (file offsets) and `t_wall_start/end`.

Sink calls happen on the capture and processing threads, serialized by one
session lock. Nothing UI-bound may run inside a sink.

```python
class MeetingSink(Protocol):
    def on_started(self, meta: MeetingMeta) -> None: ...
    def on_partial(self, text: str, label: str | None) -> None: ...
    def on_segment(self, segment: MeetingSegment) -> None: ...
    def on_stopped(self, result: MeetingResult) -> None: ...

class Diarizer(Protocol):
    def diarize(self, wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]: ...
```

Phase 1 ships `LocalMeetingSink` (JSONL writer + Library submit via an injected
callable) and **no** `Diarizer` implementation. Phase 2 plugs MOSS in as a
sliding-window batch call (local or via the server) and the session swaps the
energy labeller for diarizer output when one is configured. The future server
sink implements `MeetingSink` over tldw_server's meetings WebSocket ingest and
is registered alongside the local sink when a server is active.

### 3.4 App-owned session owner

Screens are never cached across tab switches (documented rule in `app.py`,
root-caused UI freeze 2026-07-11), so the running session cannot live on the
screen. `app.meeting_session_owner: MeetingSessionOwner` holds the current
session, exposes `prepare()` (model warm-up), `start/pause/resume/stop`, and
`async shutdown()` which is called from `_shutdown_app_owned_lifecycles`. The
owner supplies the local sink's submit callable, which marshals onto the UI
thread with `app.call_from_thread(registry.submit, ...)`. On quit the owner
stops capture and finalises files but skips the submit; the next visit to the
screen offers the recovered folder for ingest.

### 3.5 `UI/Screens/meetings_screen.py` — `MeetingsScreen`

`BaseAppScreen`, `TAB_MEETINGS` in `Constants.py`, `ScreenRoute("meetings", ...)`.

- **Rail:** mic device picker; system source picker with status line
  ("Native (macOS tap)", "Native (parec)", "Virtual device: BlackHole",
  "Unavailable, mic only"); provider readout from `console_voice_input.resolve()`
  including whether it streams word-level partials or finalises per segment;
  diarization readout ("Post-meeting speaker labels: on/off — torch/speechbrain
  missing"); one-line consent reminder; Start → Pause/Stop; timer; two level
  meters on a 200 ms `set_interval` reading `capture.levels()`.
- **Canvas:** `RichLog` of finals, `[hh:mm:ss] You: text` (label column omitted
  in room mode), plus a one-line `Static` beneath it holding the current partial
  or the "transcribing…" marker (RichLog cannot mutate its last line). After
  Stop, a footer: segment count, duration, tail-flush status, failed-segment
  count, folder path, "Open in Library".
- **Threading:** start/stop in `run_worker(..., exclusive=True, group="meeting")`;
  session listener → `call_from_thread`; the loop thread never touches the
  session directly.
- **Lifecycle:** on mount, attach to the owner's session if one is running
  (replay `session.segments` into the log, subscribe); on unmount, unsubscribe.
  Mount also runs `prepare()` in a worker and shows the existing "model
  preparing" state. Device choices persist through `save_state/restore_state`.
- **Probes on mount, in a worker:** tap probe; diarization availability via
  `importlib.util.find_spec` for `torch`, `torchaudio`, `speechbrain`, `sklearn`
  (never import `diarization_service`, whose module-level checks import torch).
- **"Open in Library":** switch to the Library route with the jobs view if the
  navigation API accepts a sub-mode (resolved in the plan); fallback is a plain
  switch plus a toast naming the job.

### 3.6 `Packaging/macos/audiotap/main.swift` — helper `tldw-audiotap`

~150 lines of Swift, macOS 14.2+ APIs:

- `CATapDescription(stereoGlobalTapButExcludeProcesses: [ownPID])` →
  `AudioHardwareCreateProcessTap` → aggregate device with the tap as sub-tap →
  `AudioDeviceCreateIOProcIDWithBlock`.
- `AVAudioConverter` to 16 kHz mono Int16.
- 2 s ring buffer filled from the IO proc; a writer thread drains it to stdout
  so a stalled reader never blocks the real-time thread; overflow drops are
  counted and reported on stderr.
- Prints `READY` on stderr once the IO proc runs; exits on stdin EOF or
  SIGTERM; exit 2 = permission denied, 3 = unsupported OS.
- Distribution: `Packaging/macos/build_app.py` compiles it into
  `Contents/MacOS/`; `Info.plist.template` gains `NSAudioCaptureUsageDescription`.
  Dev fallback: compile with `swiftc` on first use into `<data_dir>/bin/`,
  keyed by a hash of the source. If neither exists, the tap reports
  `unavailable` and the rail suggests a virtual device. Download-on-first-use
  with hash checking (Parakeet installer pattern) is a follow-up.

### 3.7 Changes to existing files

- `Audio/dictation_service_lazy.py`: `recorder_factory: Callable[..., Any] | None`
  kwarg, mirroring `RealtimeMicTap`; used by the `audio_service` property.
- `Audio/recording_service.py`: `retain_audio: bool = True` kwarg; when False,
  `_handle_audio_chunk` skips the buffer append and the queue put (otherwise
  ~230 MB/hour accumulates in the recorder alone).
- `Tests/conftest.py`: extend `_no_real_audio_device` to patch
  `AudioRecordingService._initialize_backend` to return None, so an accidental
  real recorder in a test cannot open the mic; the `real_audio_device` opt-out
  marker is unchanged.
- `Constants.py`, `UI/Navigation/screen_registry.py`, `config.py` defaults,
  `Packaging/macos/build_app.py`, `Packaging/macos/Info.plist.template`,
  `app.py` (owner + shutdown hook), `Docs/User_Guide/meetings.md`.
- No schema change. No new dependencies: numpy, sounddevice, webrtcvad are
  already in the speech extras.

## 4. Data flow

**Start.** Screen → worker → owner → session. The session creates
`<recordings_dir>/<YYYY-MM-DD_HHMM>/` (dir validated with `path_validation`),
opens `mixed.pcm` (+ `you.pcm`, `others.pcm` in call mode), `transcript.jsonl`,
`meeting.json`, then starts the dictation service; its `audio_service` property
calls the recorder factory, which returns the capture, which opens the mic and
the tap.

**Frames.** See §3.2. Both raw tracks and the mix stream to disk from the first
frame; nothing is retained in memory except the energy ring and the open speech
run.

**Attribution (call mode).** Over `[start, end]` in audio seconds the ring
yields per-source RMS buckets. A bucket is *active* for a source when
`rms > max(3 × p10_source, ABS_MIN)`, where `p10_source` is the 10th percentile of
that source's last 30 s (adaptive floor; a fixed floor mislabels noisy rooms)
and `ABS_MIN` is −60 dBFS (digital silence must not produce a zero floor).
`share_you = Σ active mic rms / (Σ active mic rms + Σ active sys rms)`;
≥ 0.7 → `you`, ≤ 0.3 → `others`, else `both`; no active buckets → higher raw
sum wins. Zoom's echo cancellation keeps your voice out of the system track,
which is what makes this work. Ceiling marked with a `ponytail:` comment
pointing at the `Diarizer` seam.

**Live output by regime.** parakeet-mlx (Apple Silicon) → streaming transcriber
→ word-level partials, 3 s context window (bounded). faster-whisper / ONNX
in-process → per-segment finals at each silence gap, hard cap 30 s per segment,
"transcribing…" marker per segment via `on_segment_transcribing`. The rail
states which regime is active. The deferred executor regime is never used by
meetings (60 s capture ceiling, `STT/dispatch_coordinator.pcm_byte_limit`).

**Pause.** Capture stops writing/feeding, drains the tap queue, timestamps stay
honest because segments carry audio offsets. **Resume** continues.

**Stop.** Tail drain → capture stop → wrap `.pcm` → `.wav` (header from byte
count) → `meeting.json` → sinks' `on_stopped` → local sink submits.

**File layout**

```
<data_dir>/meetings/2026-09-04_1430/
  mixed.wav        # always; the Library media source
  you.wav          # call mode; mic track
  others.wav       # call mode; system track
  transcript.jsonl # live segments (see record below)
  meeting.json     # schema 1: started_at, ended_at, mode, mic_device,
                   # system_source, provider, model, duration_s,
                   # segment_count, transcription_complete, failed_segments,
                   # recovered, ingest_job_id
```

JSONL record: `{"seq", "t_audio_start", "t_audio_end", "t_wall_start",
"t_wall_end", "label", "text"}`, `label` ∈ `you|others|both|null`.

Storage at PCM16 mono 16 kHz:

| Tracks kept | Per hour |
|---|---|
| mixed only | 115 MB |
| all three | 345 MB |

## 5. Library handoff

`LocalMeetingSink.on_stopped` calls the injected submit with
`source_path=<folder>/mixed.wav`, `title="Meeting <YYYY-MM-DD HH:MM>"`,
`keywords=("meeting",)`, `detected_type="audio"`,
`ingest_options={"diarization": post_diarize}` (the capability key is
`diarization`, not `diarize`). The offline pass becomes the canonical Library
transcript; the live JSONL stays in the folder as the fallback if it fails.

When `post_transcribe = false` the sink renders the live JSONL to
`transcript.md` (with the audio path in its header) and submits that as a
document instead; no re-transcription, no diarization.

When `keep_raw_tracks = false` the owner deletes `you.wav` and `others.wav`
once the registry reports the job done. `mixed.wav` is never deleted; ingest
leaves local files in place.

## 6. Configuration — `[meetings]`

| Key | Default | Meaning |
|---|---|---|
| `provider` | `"auto"` | STT provider; `auto` = `console_voice_input.resolve()` |
| `model` | `""` | provider model override |
| `system_source` | `"auto"` | `auto` (native probe) or an input-device name (virtual device) |
| `mic_device` | `""` | input device name; empty = default |
| `recordings_dir` | `<data_dir>/meetings` | validated with `path_validation` |
| `keep_raw_tracks` | `true` | keep `you.wav` / `others.wav` after ingest |
| `post_transcribe` | `true` | run the offline pass on `mixed.wav` |
| `post_diarize` | `true` | request diarization in that pass |

Read through `get_cli_setting("meetings", key, default)` (flat section; the
dotted-key lookup is known-broken for nested sections).

## 7. Error handling and recovery

**At start**
- Mic will not open → existing macOS permission copy from the dictation
  service; Start stays enabled.
- System source unavailable (permission denied, helper missing, macOS < 14.2,
  no `parec`/`pw-record`) → session degrades to room mode; rail says why.
- Model fails to load → error state on the rail; nothing recorded.

**Mid-meeting**
- Helper/`parec` dies → zero-fill, rail "System source lost", one restart after
  2 s, then stays lost; the meeting continues from the mic.
- Mic disappears → recorder error callback ends the session cleanly (the mic is
  the clock); files finalise; handoff still runs.
- Segment transcription error → the service drops it; the screen keeps a
  failed-segment count in the footer.
- Disk write error → session stops with an error, keeps what is on disk.

**Crash safety.** The stdlib `wave` module writes its header only on close, so
raw PCM goes to `.pcm` and is wrapped on stop. On the next screen mount any
folder with a leftover `.pcm` is wrapped the same way, `meeting.json` gets
`ended_at` from file size and `recovered: true`, and the footer offers "Ingest
recovered meeting".

**Stop.** If the dictation result reports `transcription_complete == False` the
footer says the last segment was dropped. If the registry refuses the submission
(`ActiveIngestSubmissionRefused`) the footer says "saved locally, not queued"
with the folder path.

**Contention.** Console dictation reports busy while a meeting runs (the mic is
held). Expected; no special handling.

## 8. Testing

All hardware-free under the (extended) autouse guard; no test constructs a real
recorder or tap.

- **Capture:** two fake sources (tone vs silence, then injected room noise).
  Saturating mix, zero-fill and backlog drop keep skew ≤ 200 ms, bytes per
  track, speech runs open/close at the silence threshold, labels you/others/both
  including the adaptive floor. Hypothesis property: random arrival patterns
  never overflow int16 or exceed the skew bound.
- **Session:** fake dictation service driven through its callbacks. Segment
  audio/wall offsets and labels, FIFO run consumption incl. the no-closed-run
  case, JSONL contents, sink call order, submit kwargs (`diarization` key),
  Markdown path when `post_transcribe` is off, commands disabled and
  auto-clear forced on the built service, in-process facade (no dispatcher).
- **Tap:** a Python stand-in helper emitting PCM to stdout, and a variant that
  dies mid-stream: frame delivery, restart-once, lost state, queue overflow
  drops oldest. Platform resolvers are pure functions over fake device lists
  (`[Loopback]` matching, virtual-device names, `pactl` output parsing).
- **Recorder flag:** `retain_audio=False` retains nothing and still invokes the
  callback.
- **Recovery:** leftover `.pcm` → valid WAV header, `meeting.json` updated.
- **Owner:** shutdown finalises files and skips submit; submit callable marshals
  through `call_from_thread`.
- **Screen:** a few `app.run_test()` pilots with the owner faked: start → pause →
  stop transitions, rows with and without labels, partial line updates, footer
  after stop, attach-on-mount replays existing segments.
- **Swift helper:** opt-in marker test on macOS that compiles it and checks
  frames arrive; not run in CI.
- **Live verification** (per `backlog/docs/lessons-live-verification.md`):
  drive the TUI under tmux while a video plays as system audio and the mic
  hears the room; record the You/Others evidence and a Library ingest result in
  the task's Implementation Notes.

## 9. Follow-ups (not in phase 1)

1. **Phase 2 — live speaker labels:** `Diarizer` implementation over MOSS
   Transcribe-Diarize (0.9B, batch, 30 s Whisper-feature windows, CUDA-first,
   Apache-2.0) as a sliding-window call, most likely hosted on tldw_server;
   swap the energy labeller for its output. Needs its own design.
2. **Server sink:** `MeetingSink` over tldw_server's meetings WebSocket ingest
   (the adapter gap `Meetings_Interop` already reports), creating the session
   via the existing REST wrappers and finalising on stop.
3. **Windows loopback verification** on a real Windows box.
4. **Helper download-on-first-use** with hash check for source installs on
   non-dev Macs.
5. **Rolling deferred captures** in the STT executor, if in-process ONNX for
   an hour proves a problem on low-memory machines.
6. **Zoom recording import** (M4A/VTT with per-participant attribution) as a
   post-hoc alternative.

## 10. Assumptions to confirm during implementation

- sounddevice's bundled PortAudio on Windows enumerates WASAPI loopback devices
  with the `[Loopback]` suffix (PortAudio ≥ 19.7).
- The Library navigation API can open the jobs view directly; otherwise the
  fallback in §3.5 applies.
- `pactl` is present wherever `parec` is (both ship with pulseaudio-utils /
  pipewire-pulse).
