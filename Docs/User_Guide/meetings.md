# Meetings — record a call or a room with a live labelled transcript

## What this screen is for

Meetings records a live conversation — a video call (mic + the other
participants' system audio) or an in-person room (mic only) — and shows a
live transcript while it records. Stopping a meeting saves the raw audio and
a segment-by-segment transcript to disk and queues the recording for Library
ingest with diarization, so the finished meeting shows up as a searchable
media item alongside everything else in your Library.

## Getting there

Press **F11**, click **F11 Meetings** in the nav bar, or open the command
palette (**Ctrl+P**) and choose **"Tab Navigation: Switch to Meetings"**.
There is no digit hotkey — Meetings sits past the Ctrl+0 row alongside Lab,
Logs, Settings, and Research.

## Layout tour

The screen is a two-pane workbench under a one-line purpose banner:

- **Sources** (left pane) — two device pickers (microphone, system-audio
  source), three status lines, a consent note, the **Start / Pause / Stop**
  row, a timer, two audio-level meters, a recovery line, and a **Recover**
  button.
- Transcript canvas (right pane) — the live transcript log, a partial-line
  indicator while a segment is still being transcribed, a footer line that
  appears after Stop, and an **Open in Library** button.

## Features & controls

| Control | What it does |
|---|---|
| Microphone picker | Choose the input device for your own voice. Defaults to "System default"; other entries are named exactly as your OS reports them (e.g. "MacBook Pro Microphone"). |
| System-audio picker | Choose how the other participants' audio is captured: "Native (auto)" probes the OS for a built-in tap; any other entry is a specific input device name (see "Virtual-device fallback" below). |
| **System audio: …** status line | What the system-audio picker resolved to for this session — a native tap ("Native (macOS tap)" / "Native (parec)"), a named virtual device ("Virtual device: BlackHole"), or "Unavailable, mic only (…)" with the reason, in which case the meeting records room-mode (mic only). |
| **Transcriber: …** status line | The speech-to-text provider and model in use, plus "(finalises per segment)" — each transcript row is a *final* for its own segment, not a running partial for the whole meeting. |
| **Speaker labels after the meeting: …** status line | Whether offline diarization will run once you stop: "on", or "off (…)" naming the missing Python packages (see "Speaker labels" below). |
| Consent note | "Recording other people may require their consent." — a static reminder, not a gate; the app does not ask anyone else for consent on your behalf. |
| **Start** | Begins recording and live transcription. Disabled while a meeting is already running or before the device probe finishes. |
| **Pause** / **Resume** | Pauses capture and transcription in place; the same button relabels itself and resumes where it left off. |
| **Stop** | Ends the meeting, finalizes the audio files and transcript, and queues the Library ingest job. |
| Timer | Elapsed recording time as `HH:MM:SS`, updated roughly 5×/second while recording. |
| Level meters | Two bars (mic, system) showing live input level, 0–100%. |
| **Recover** | Appears enabled with a line reading "Unfinished meeting found: `<folder-name>`" when the screen finds a meeting folder left behind by a crash or forced quit. Recovering patches the audio files, marks the meeting `recovered` in its metadata, and queues it for Library ingest — same as a normal Stop. |
| **Open in Library** | Enabled once a meeting has been queued for ingest (by Stop or Recover); switches to Library's Import view with the ingest queue in view. |

## Common tasks

1. **Record an in-person conversation (room mode).** Open Meetings, confirm
   the "System audio" line says "Unavailable, mic only (…)" or leave the
   system-audio picker on a virtual device you don't have — either way the
   meeting still runs mic-only. Press **Start**, talk, press **Stop** when
   done.
2. **Record a call (mic + the other side).** On a Mac with the System Audio
   Recording permission already granted to your terminal, confirm the
   status line reads "Native (macOS tap)" (or "Native (parec)" on Linux),
   press **Start**, hold the call, press **Stop**.
3. **Switch input devices before you start.** Open the microphone or
   system-audio picker and choose a different entry; the app remembers your
   choice for next time. Changing the system-audio picker re-runs the
   device probe and status lines refresh.
4. **Pause during a meeting.** Press **Pause** to stop recording and
   transcribing without ending the session; press the same button (now
   labelled **Resume**) to continue.
5. **Recover a meeting after a crash or a forced quit.** Reopen Meetings —
   if a folder was left unfinished, the recovery line and an enabled
   **Recover** button appear automatically. Press **Recover**; the footer
   reports the outcome ("Recovered `<folder>`: Library ingest queued:
   `<job-id>`." or "…, saved locally, not queued (`<reason>`).").
6. **Find the finished recording in your Library.** After Stop (or
   Recover), press **Open in Library** to jump straight to the Import
   view's ingest queue, or open Library yourself later — the media item
   carries the meeting's title, the `meeting` keyword, and (once processed)
   its transcript.

## Keyboard & commands

| Key | Action |
|---|---|
| F11 | Switch to Meetings (also reachable via the command palette; no other screen-specific bindings) |

All other actions on this screen (device pickers, Start/Pause/Stop,
Recover, Open in Library) are mouse/pointer controls with no dedicated key
binding; use Tab to move focus between them.

## Related settings & docs

Meetings reads a `[meetings]` section in `config.toml` (`get_cli_setting`,
flat keys only — a dotted lookup into a nested table does not work here):

| Key | Default | Meaning |
|---|---|---|
| `provider` | `"auto"` | Speech-to-text provider; `"auto"` resolves the same way Console dictation does. |
| `model` | `""` | Provider model override. |
| `system_source` | `"auto"` | `"auto"` runs the native-tap probe; any other value is an input-device name (for a virtual-cable setup). |
| `mic_device` | `""` | Input device name for your own voice; empty uses the system default. |
| `recordings_dir` | `<data_dir>/meetings` | Where meeting folders are written. |
| `keep_raw_tracks` | `true` | Keep the separate `you.wav` / `others.wav` files after Library ingest finishes (rather than deleting them once the raw-track cleanup runs). |
| `post_transcribe` | `true` | Run the offline transcription pass on `mixed.wav` during Library ingest. |
| `post_diarize` | `true` | Ask that offline pass to also diarize (assign speaker labels) — see "Speaker labels" below. |

Each finished meeting's folder (named by start time, e.g.
`2026-09-04_2121/`) contains:

- `mixed.wav` — the combined recording (what gets ingested into Library).
- `you.wav` / `others.wav` — the separate raw tracks (kept or deleted per
  `keep_raw_tracks`).
- `transcript.jsonl` — one JSON object per finalized segment.
- `meeting.json` — session metadata: start/end time, duration, mode,
  device/provider choices, segment and failure counts, the Library ingest
  job id, and (after a crash) `recovered: true`.

See also: [Library ▸ Import & export](library/import-and-export.md) for
what happens to a meeting once it's queued.

## Quirks & troubleshooting

- **macOS System Audio Recording permission.** Capturing the other side of
  a call needs macOS's System Audio Recording permission granted to
  whatever terminal/app runs tldw_chatbook (macOS prompts the first time a
  session actually tries to tap system audio). Until it's granted, the
  helper process exits immediately and the session falls back to mic-only
  ("room mode") with all rows labelled "You" — Meetings does not ask for
  this permission proactively, and there is no in-app control to request
  it early. Packaged macOS builds declare
  `NSAudioCaptureUsageDescription` ("tldw_chatbook records what your
  computer plays so meetings can be transcribed.") so the OS prompt shows
  useful text.
- **"System audio: Native (macOS tap)" during recording, with no visible
  change when the tap actually fails.** On a host without the permission
  above, starting a call-mode meeting spawns the native helper, which exits
  immediately (permission denied), restarts once automatically, and then
  gives up; per the design the session should continue as room-mode with a
  degraded status ("System source lost"). In live testing on a host without
  the permission granted, the rail's status line kept reading "System
  audio: Native (macOS tap)" for the whole session with no visible change,
  and `meeting.json` likewise recorded `"mode": "call"` /
  `"system_source": "Native (macOS tap)"` even though no system audio was
  actually captured (the system track was silent). The meeting itself
  still finalizes correctly and the mic track is unaffected — this is a
  cosmetic gap in the "source lost" indicator, flagged for the
  whole-branch review rather than fixed here, not a data-loss issue.
- **Virtual-device fallback.** On a host without native system-audio
  support (macOS below 14.2, or no `parec`/`pw-record` on Linux), install a
  loopback device — [BlackHole](https://existential.audio/blackhole/) on
  macOS or [VB-Cable](https://vb-audio.com/Cable/) on Windows — and pick it
  from the system-audio device dropdown instead of "Native (auto)".
- **Speaker labels ("who said what") are computed after the meeting, not
  live.** The live transcript during recording only ever distinguishes
  "You" from "Others" (or omits labels entirely in room mode); per-speaker
  diarization runs as part of the offline Library ingest pass, and only
  when `torch`, `torchaudio`, `speechbrain`, and `scikit-learn` are
  installed — otherwise the "Speaker labels after the meeting" status line
  reads "off" and names the missing packages, and diarization is simply
  skipped.
- **Each transcript row is a per-segment final, not a whole-meeting
  transcript.** Rows can lag live speech by up to roughly the length of one
  segment (up to ~10 seconds) plus however long that segment took to
  transcribe — there is no cross-segment "live partial" merge.
- **Windows loopback capture is unverified.** The design calls for
  `sounddevice` to enumerate a `[Loopback]` WASAPI device automatically;
  this has not been confirmed on a real Windows machine.
- **Call mode (native system-audio tap) is unverified end-to-end on this
  host.** The live verification for this page ran without the macOS System
  Audio Recording grant, so only room mode (mic-only) was confirmed to
  produce a working recording, footer, and Library handoff. To verify call
  mode yourself: grant System Audio Recording to your terminal app once
  (macOS prompts on first use), run
  `TLDW_RUN_AUDIOTAP_HELPER_TEST=1 .venv/bin/python -m pytest Tests/Audio/test_audiotap_helper_macos.py -p no:cacheprovider`,
  then start a real call.
- **No speech content in a silent room produces zero transcript rows and
  "failed segment(s)" in the footer**, not an error state — the
  speech-to-text pass simply has nothing to transcribe. This is expected,
  not a bug.
- **The queued Library ingest job may sit at "queued" rather than
  progressing to "done"** if your install lacks the optional audio
  transcription dependencies (e.g. `faster-whisper`); the meeting's own
  files and metadata are unaffected either way.

—
*Verified against dev @ ac6c511cb6 — 2026-09-04*
