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
  source), four status lines, a consent note, the **Start / Pause / Stop**
  row, a timer, two audio-level meters, a recovery line, and a **Recover**
  button.
- Transcript canvas (right pane) — a **Speakers** legend that grows one row
  per distinct speaker as the meeting identifies them, the live transcript
  log, a partial-line indicator while a segment is still being transcribed,
  a footer line that appears after Stop, and an **Open in Library** button.

## Features & controls

| Control | What it does |
|---|---|
| Microphone picker | Choose the input device for your own voice. Defaults to "System default"; other entries are named exactly as your OS reports them (e.g. "MacBook Pro Microphone"). |
| System-audio picker | Choose how the other participants' audio is captured: "Native (auto)" probes the OS for a built-in tap; any other entry is a specific input device name (see "Virtual-device fallback" below). |
| **System audio: …** status line | What the system-audio picker resolved to for this session — a native tap ("Native (macOS tap)" / "Native (parec)"), a named virtual device ("Virtual device: BlackHole"), or "Unavailable, mic only (…)" with the reason, in which case the meeting records room-mode (mic only). |
| **Transcriber: …** status line | The speech-to-text provider and model in use, plus "(finalises per segment)" — each transcript row is a *final* for its own segment, not a running partial for the whole meeting. |
| **Speaker labels after the meeting: …** status line | Whether offline diarization will run once you stop: "on", or "off (…)" naming the missing Python packages (see "Speaker labels" below). |
| **Live speaker labels: …** status line | Whether speaker ids will be assigned *while* recording (what fills the Speakers legend): "on (ONNX)" / "on (SpeechBrain)" naming the engine, or "off (…)" with the reason — "not enabled in settings" (`meetings.live_diarization` is off), the missing packages, or an unsupported `diarizer_backend`. While a meeting runs it also reports the ONNX engine's first-Start model download ("downloading 12 / 44 MB", "warming up") — see "Speaker-label engines and their models" below. |
| Consent note | "Recording other people may require their consent." — a static reminder, not a gate; the app does not ask anyone else for consent on your behalf. |
| **Start** | Begins recording and live transcription. Disabled while a meeting is already running or before the device probe finishes. |
| **Pause** / **Resume** | Pauses capture and transcription in place; the same button relabels itself and resumes where it left off. |
| **Stop** | Ends the meeting, finalizes the audio files and transcript, and queues the Library ingest job. |
| Timer | Elapsed recording time as `HH:MM:SS`, updated roughly 5×/second while recording. |
| Level meters | Two bars (mic, system) showing live input level, 0–100%. |
| **Recover** | Appears enabled with a line reading "Unfinished meeting found: `<folder-name>`" when the screen finds a meeting folder left behind by a crash or forced quit. Recovering patches the audio files, marks the meeting `recovered` in its metadata, and queues it for Library ingest — same as a normal Stop. |
| **Speakers** legend | One row per speaker identified so far ("Speaker 1", "Speaker 2", … until renamed), each with a rename box next to it. Only fills in when a backend is actually assigning speaker ids to segments as they arrive (see "Speaker labels" below) — otherwise it stays empty and every row keeps reading "You"/"Others". |
| Speaker rename box | Type a name and press Enter to rename that speaker everywhere: the legend row and every already-shown transcript line for that speaker, and it's saved into the meeting's `meeting.json` name map immediately (not only at Stop). Clearing the box and pressing Enter reverts that speaker back to the generic "Speaker N" label. |
| Footer line (after Stop) | What was saved and where it went: segment count and duration, any dropped/failed segments, the folder, and the Library ingest job id (or why none was queued). It also reports "Speaker labels unavailable (…)" when live labelling was wanted but the backend never started or crashed mid-meeting (the recording, transcript and ingest are unaffected — the meeting simply stays on "You"/"Others"), and "Speaker merge to resolve: Alice / Bob" when the final pass decided two speakers you had named separately are one person; both names are kept on the survivor for you to fix. |
| **Open in Library** | Enabled once a meeting has been queued for ingest (by Stop or Recover); switches to Library's Import view with the ingest queue in view. |

### Remember my voice

The rail's **Your voice** section (below Recover; scroll down on a small
terminal — see "The rail scrolls" below) lets you enroll your own voice once
so later meetings tag your speech with your display name automatically,
including room mode and hybrid rooms where the mic channel isn't a clean
"you". There is no Settings › Meetings page for this yet — every control
lives here, on the Meetings rail, and the two on/off choices below are
config keys, not switches on this screen.

- **Enroll my voice.** Records about 30 seconds from your microphone
  (countdown shown, with a **Cancel** button) and turns it into one voice
  embedding. Refused with "The microphone is busy — stop the meeting
  first." while a meeting is running, and **Start** is refused in turn while
  an enrollment is in progress. The sample audio itself is never written to
  disk — only the resulting embedding is saved.
- **Voice match: …** rail line, under the other status lines and just above
  the Start/Pause/Stop row, reports whether this
  meeting can tag you by voice and why not: `on`, or `off` with one of
  `disabled` (the `voice_match` setting), `plain call mode` (see below),
  `no voiceprint`, `needs re-enrollment` (the voiceprint was made with a
  different embedding model), `live speaker labels off` (see below),
  `store unreadable`, `keyring locked`, or
  `store unavailable`. Before Start this line is provisional — it only
  checked that a record exists, not that it can be decrypted — so re-check
  it once the meeting is recording.
- **The matched speaker gets a small marker** (` ·`) next to their name, in
  both the Speakers legend and the transcript rows, so you can tell an
  automatic match from a name you typed yourself. Renaming that speaker to
  anything other than your display name is treated as an override and
  clears the marker; renaming it back to your display name is not an
  override.
- **Voice match needs live speaker labels on.** Matching compares the live
  speaker clusters against your voiceprint, and those only exist when a live
  diarizer is running — so with `[meetings] live_diarization = false` (the
  default) the rail reads "Voice match: off (live speaker labels off)" and
  nothing is ever tagged. Set `live_diarization = true` (with either engine
  available — see "Speaker-label engines and their models" below) to turn
  matching on.
- **Plain call mode never matches.** In a call with `diarize_mic_channel`
  off (the default), the mic channel is already assumed to be you, so voice
  matching never runs there — there's nothing to disambiguate. It runs in
  room mode always, and in call mode only when `diarize_mic_channel` is on
  (see "Hybrid rooms" above).
- **Learning offer.** After a meeting that matched you by voice — or after a
  plain call-mode meeting (`diarize_mic_channel` off) that kept its
  `you.wav` track, where the mic channel is assumed to be you — a rail
  prompt appears — never a popup, nothing
  blocks — asking "Remember this voice as yours for future meetings?" (or,
  when the sample came from the plain mic channel, "Was it only you on the
  mic?"). **Accept** blends that meeting's sample — the matched cluster, or
  in plain call mode the first minute of the meeting's own `you.wav` — into
  your stored
  voiceprint; **Not now** keeps nothing and asks again next time; **Don't
  ask again** keeps nothing and turns the `voice_learn_offer` setting off
  for good. At most one offer appears per meeting, and an offer you never
  answer — because you switched tabs or started another meeting — is
  dropped rather than left pending indefinitely. An offer only ever appears
  once you already have a stored voiceprint: **Enroll my voice** is always
  the first step, and learning only refines what is already there. Accepting
  can take a moment on a cold model — the row says "Warming up the voice
  model…" while it waits.
- **Delete / Export… / Import…**, on the row below Enroll. **Delete** is a two-step
  confirm ("Press Delete again to remove your stored voiceprint.") — once
  gone, it cannot be recovered. **Export…** opens a form for a passphrase
  and a destination file path and writes an encrypted copy there —
  the passphrase is never shown, logged, or echoed back. **Import…** opens
  the same form with **Merge** (blend the file's voiceprint into the one
  stored here — only *accepted* when the file was made with the same
  embedding model, and with a vector of the same size) and **Replace**
  (discard what's stored and keep the
  file's instead — the only option when the model differs, reported as
  "Different model — choose Replace"). A passphrase that does not open the
  file is reported as "Wrong passphrase", and exporting *onto* your own
  stored voiceprint is refused rather than destroying it. The typed path is
  checked before anything is read or written: an import must name a file that
  is actually there ("Import needs a file that is there to read."), and an
  export must name a plain file in a folder that exists ("Export needs a
  plain file in a folder that exists.") — a symlinked destination is refused,
  so an export can never be redirected somewhere you did not name. A file
  that opens with the right passphrase but does not hold a usable voiceprint
  is refused too ("Import file is not a valid voiceprint"), leaving what you
  already have untouched.
- **Encryption.** The voiceprint is one JSON record encrypted with its own
  random key, stored at `voiceprint.json` in the app's user data directory
  (not the configurable `recordings_dir`, which may be a synced folder).
  The key lives in your OS keyring when one is available ("Voice: enrolled
  (keyring)"); otherwise it falls back to an owner-only key file in the
  same directory ("Voice: enrolled (key file)") — a key file with looser
  permissions than that is refused rather than used. Reading the key at
  meeting Start is bounded to 1.5 seconds on a worker thread, so a
  locked or slow keyring never delays Start; that meeting simply runs with
  "Voice match: off (keyring locked)" and the next Start tries again.
  Opening the Meetings screen itself never touches the key — it only asks
  which mode is in force, to show "Voice: enrolled (keyring)" or "(key
  file)". The key is *read* at meeting Start (above, bounded, on a worker
  thread) and *created* only by explicit enrollment, which is why a macOS
  Keychain prompt (if any) shows up there and not when you just open the
  tab. A key file whose permissions have been widened is refused outright —
  the rail then reads "Voice match: off (store unavailable)" and the repair
  is `chmod 600`, not unlocking anything.
- **Privacy.** What's stored is one averaged voice embedding (a list of
  numbers) plus a sample count and timestamps — never raw audio, never a
  transcript, and never anything from a specific meeting beyond that
  running average. The embedding never appears in logs, `meeting.json`, or
  a meeting's exported/ingested files; it crosses only the local pipe to
  the diarization worker process, on this device. Deleting the voiceprint
  removes that one file; nothing else references it.
- **The match threshold is a starting value, not a validated one.**
  `voice_match_threshold` (default `0.2`, a cosine-distance ceiling — lower
  is stricter) hasn't been calibrated against real speech on every install;
  raise it if you're never matched, lower it if someone else gets matched
  as you. There is no on-screen "how close was your last match" readout
  yet — the field exists in the stored record but nothing computes it. The
  bake-off (linked below) does measure one per engine on its own corpus, and
  its numbers say `0.2` is stricter than it needs to be: roughly `0.5` for
  SpeechBrain, roughly `0.38` for the ONNX `titanet_small` embedder. Treat
  those as better starting points than the shipped default, not as calibrated
  for your voice. The closest thing to a number for *your* voice today is the
  opt-in test at
  `Tests/Audio/test_voiceprint_real.py`, which enrolls a real voice with the
  macOS `say` command and prints the similarity it measured.
- **The rail scrolls.** The Meetings rail is a scrolling pane, so on a
  small terminal (roughly 100×30 and below) the Voice row and the learning
  offer can sit below the fold — scroll the rail down to reach them.

### Speaker-label engines and their models

Live speaker labels (`meetings.live_diarization = true`) come from one of two
engines, chosen by `meetings.diarizer_backend`:

| `diarizer_backend` | What runs | What it needs installed |
|---|---|---|
| `"auto"` (default) | SpeechBrain if its packages are there, otherwise ONNX | either row below |
| `"onnx"` | sherpa-onnx, plus two ONNX model files fetched on the first Start | `sherpa_onnx` and `numpy` — both are in the base install, no torch |
| `"speechbrain"` | SpeechBrain's ECAPA model on torch | the `diarization` extra: `torch`, `torchaudio`, `speechbrain`, `sklearn` |
| `"server"` | nothing — reserved; the rail reads "off (unsupported backend: server)" | — |

`"local"`, the value this page used to document as the only one, is still
accepted and now means `"auto"`.

The "off (missing: …)" rail line names **import** names, which are not always
what you type to install them:

| The rail says missing | Install with |
|---|---|
| `sherpa_onnx`, `numpy` | `pip install sherpa-onnx numpy` |
| `torch`, `torchaudio`, `speechbrain`, `sklearn` | `pip install -e ".[diarization]"` |

That list only appears when you pinned an engine explicitly. Under `"auto"`
the line reads "off (install the diarization extra)" instead, because a
missing-package list for whichever engine happened to be tried last is not
advice you asked for.

**The ONNX engine downloads its models the first time you press Start** —
from the sherpa-onnx GitHub releases into
`<user data dir>/models/diarization/onnx/`. Two files: the shared
segmentation model, plus the one embedder your `onnx_embedder` names.

| File | sha256 (first 12) | Size |
|---|---|---|
| `pyannote-segmentation-3-0.onnx` (always) | `220ad67ca923` | 6.0 MB, fetched inside a 7.0 MB tarball |
| `nemo_en_titanet_small.onnx` (`titanet_small`, the default) | `ad4a1802485d` | 40.3 MB |
| `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx` (`eres2net_en`) | `c59158379255` | 26.5 MB |
| `wespeaker_en_voxceleb_resnet34.onnx` (`wespeaker_resnet34`) | `5ef208a9da14` | 26.5 MB |
| `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` (`campplus_en`) | `357a834f702b` | 29.6 MB |

Each file is verified against that hash when it lands and again every time
the engine loads it; a bad download is discarded and retried once. Traffic
only ever goes to `github.com` (redirects are followed only to
`*.githubusercontent.com`), the whole fetch is bounded to 10 minutes, and
nothing about your meeting is uploaded — the model fetch is the only network
traffic Meetings makes.

The "Live speaker labels:" rail line reports the whole sequence:

| Line | What it means |
|---|---|
| `on (ONNX, ~47 MB of models fetched at Start)` | the models are not on disk yet; the next Start fetches them (the number is for `titanet_small`; ~33 MB for the two 26 MB embedders, ~37 MB for `campplus_en`) |
| `downloading 12 / 44 MB` | the fetch, updated once per MB — its total counts the model files' own bytes, so it is a little under the estimate above, which counts the tarball actually transferred |
| `warming up` | files in place; the worker process is loading them |
| `on (ONNX)` / `on (SpeechBrain)` | labelling |
| `off (models unavailable)` | the ONNX model fetch or load failed — recording, transcript and Library ingest are unaffected; the meeting simply keeps "You"/"Others" |
| `off (backend unavailable)` | the labelling process itself failed (it never finished loading, or it crashed twice) — nothing to download, and the same "recording is unaffected" applies. Common on the SpeechBrain engine, where the models are already on disk |
| `off (not enabled in settings)` | `live_diarization` is `false` (the default) |

Recording never waits for any of this: the download runs on its own thread
and Start returns immediately, so a meeting that starts during the fetch
records normally and picks up speaker ids when the engine is ready.

- **`onnx_models_dir` is where the models live**, not just where they are
  downloaded to. Leave it empty for the folder above. Point it at a folder of
  files you placed there yourself for an air-gapped machine: a file already
  there is hash-verified and never re-fetched — but a file that is simply
  *absent* is still downloaded, so an offline install must pre-place **both**
  the segmentation file and its embedder. A pre-placed file whose hash does
  not match is refused outright and never re-fetched (the rail then reads
  "off (models unavailable)").
- **`onnx_embedder` picks the ONNX voice model**: `titanet_small` (default),
  `eres2net_en`, `wespeaker_resnet34`, or `campplus_en`. It is part of your
  voiceprint's identity, so changing it means enrolling your voice again.
- **`auto` means ONNX.** It re-decides on every Start and tries ONNX first;
  since sherpa-onnx ships with the base install, `auto` only ever falls back
  to SpeechBrain when the ONNX package itself is broken. To use SpeechBrain
  set `diarizer_backend = "speechbrain"` explicitly. The two engines produce
  different kinds of vector, so an enrolled voiceprint made with the other
  one reports "Voice match: off (needs re-enrollment)" until you enroll again
  — a voiceprint you enrolled before this release, when SpeechBrain was the
  engine, will ask for that once.

**Which engine is better?** They were measured against each other over 188
minutes of labelled conversation (VoxConverse + AMI) — see
[the bake-off report](../STT_Evaluation/task-31827/report.md). ONNX wins the
after-the-meeting accuracy: `titanet_small`, the default embedder, scores a
diarization error of 0.143 against SpeechBrain's 0.371, and `eres2net_en` is
the best of the four at 0.125. It also wins the live-labelling purity (1.000
for `titanet_small`, against 0.941) and the per-window latency (11.8 ms for
`titanet_small`, against 27.6 ms), and it needs no torch at all. What it does
*not* win is voice-match separation — how far your enrolled voice sits from
everyone else's — where SpeechBrain's margin is 0.734 and the best ONNX
embedder's (`titanet_small` again) is 0.640. That margin compares two
different kinds of vector on one absolute scale, and both leave the shipped
`voice_match_threshold` of 0.2 comfortably inside them (your own voice sits
at a distance of about 0.07 from its ONNX voiceprint, everyone else at about
0.70), so ONNX is the default engine. Set `diarizer_backend = "speechbrain"`
if you want the wider margin and have the `diarization` extra.

**Model attribution.** `nemo_en_titanet_small.onnx` is NVIDIA's TitaNet-small,
licensed **CC-BY-4.0** (© NVIDIA), redistributed through the sherpa-onnx
releases. The other assets: pyannote `segmentation-3.0` (MIT), WeSpeaker
ResNet34 (Apache-2.0), and 3D-Speaker ERes2Net / CAM++ (Apache-2.0; check
ModelScope's own terms before redistributing them).

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
| `live_diarization` | `false` | Assign speaker ids while recording instead of only in the offline pass — feeds the Speakers legend. Needs one of the two engines below; the base install already ships the ONNX one. |
| `diarizer_backend` | `"auto"` | Which engine assigns the live speaker ids: `"auto"`, `"onnx"`, `"speechbrain"`, or the reserved `"server"` — see "Speaker-label engines and their models" above. `"local"` still works and means `"auto"`. |
| `onnx_embedder` | `"titanet_small"` | Which voice model the ONNX engine uses: `titanet_small`, `eres2net_en`, `wespeaker_resnet34`, `campplus_en`. Part of your voiceprint's identity — changing it means enrolling again. |
| `onnx_models_dir` | `""` (→ `<data_dir>/models/diarization/onnx`) | Where the ONNX model files live. Point it at pre-placed files for an air-gapped install; anything missing there is still downloaded. |
| `max_speakers` | `8` | Upper bound the local live diarizer uses when clustering voices into speaker ids. |
| `diarize_mic_channel` | `false` | Hybrid rooms: also diarize the mic ("you") and overlap ("both") channels in call mode instead of always pre-naming them — see "Speaker labels" below. |
| `voice_match` | `true` | Tag your own speech with your display name using an enrolled voiceprint — see "Remember my voice" above. No effect until you enroll one. |
| `voice_match_threshold` | `0.2` | Cosine-distance ceiling for calling a cluster you (lower = stricter). A starting value, not validated for your voice/model — see "Remember my voice" above. |
| `voice_match_min_seconds` | `4` | Seconds of your speech a cluster must accumulate before it can be matched at all. |
| `voice_learn_offer` | `true` | Offer to learn from a meeting that matched you cleanly (at most once per meeting) — "Don't ask again" in that offer also flips this to `false`. |

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
  helper process exits immediately and the session is designed to fall
  back to mic-only ("room mode") — Meetings does not ask for this
  permission proactively, and there is no in-app control to request it
  early. Packaged macOS builds declare
  `NSAudioCaptureUsageDescription` ("tldw_chatbook records what your
  computer plays so meetings can be transcribed.") so the OS prompt shows
  useful text.
- **`meeting.json` records the system source you started with, not the one
  you ended with.** On a host without the permission above, starting a
  call-mode meeting spawns the native helper, which exits immediately
  (permission denied), restarts once automatically, and then gives up. The
  rail notices: its status line flips to "System audio: System source lost
  — continuing from the microphone", and the meeting carries on mic-only.
  What does *not* change is the metadata — `meeting.json` keeps the
  `"mode": "call"` / `"system_source": "Native (macOS tap)"` it was opened
  with, because those fields record the start-time choice. So a recording
  whose system track is silent can still be labelled `call` in its
  metadata. The meeting finalizes correctly and the mic track is
  unaffected either way; this is a metadata nuance, not a data-loss issue.
- **Virtual-device fallback.** On a host without native system-audio
  support (macOS below 14.2, or no `parec`/`pw-record` on Linux), install a
  loopback device — [BlackHole](https://existential.audio/blackhole/) on
  macOS or [VB-Cable](https://vb-audio.com/Cable/) on Windows — and pick it
  from the system-audio device dropdown instead of "Native (auto)".
- **Speaker labels ("who said what") are computed after the meeting by
  default, not live.** With `meetings.live_diarization` left at its default
  (off), the live transcript during recording only ever distinguishes "You"
  from "Others" (or omits labels entirely in room mode), the Speakers legend
  stays empty, and per-speaker diarization runs once, as part of the offline
  Library ingest pass — only when `torch`, `torchaudio`, `speechbrain`, and
  `scikit-learn` are installed (install them together with the
  `diarization` extra: `pip install -e ".[diarization]"`); otherwise the
  "Speaker labels after the meeting" status line reads "off" and names the
  missing packages, and diarization is simply skipped. (That offline pass is
  torch-only — the ONNX engine described in "Speaker-label engines and their
  models" above is for the *live* labels, not for it.) Turning on
  `meetings.live_diarization` (with `diarizer_backend` left at its default,
  `"auto"`, and either engine's packages installed) assigns speaker ids as
  segments arrive instead: the legend fills in with "Speaker 1", "Speaker
  2", … as each new voice is heard, and typing a name into a row's rename
  box relabels that speaker everywhere — the legend, the transcript shown so
  far, and the name map saved into `meeting.json` for the finished
  recording. This live path has automated pilot-test coverage only; it has
  not been exercised in a live session on this page's verification host.
- **The name that stands in for you is fixed when the meeting starts.** It
  is your configured chat display name when you have set one (otherwise
  "You"), and it is stamped onto the recording at Start — so changing that
  setting while a meeting is running leaves the rows already on screen, the
  rows still to come, and the saved transcript all saying the same thing.
  The new value applies to the next meeting you start.
- **Hybrid rooms — someone else sharing your microphone — can also be
  diarized, behind `meetings.diarize_mic_channel` (off by default).**
  Normally the mic ("you") and overlap ("both") channels in call mode are
  never sent through the diarizer: every mic-channel segment is pre-named as
  you regardless of who is actually speaking into that mic. Turning this flag
  on (in addition to `live_diarization`) sends those channels through the
  diarizer too, so a "you"/"both" segment that gets a speaker id renders by
  that id (a name, or "Speaker N") instead of your display name — the "You"
  pre-naming no longer applies once a segment has been diarized. This only
  changes call mode; room mode already diarizes every segment regardless of
  this flag.
- **Speaker names can also be renamed after the fact, on the finished
  Library item — not only live, during the meeting.** Open the recording in
  Library ▸ Media and scroll to the bottom of the Read tab: a **Rename
  speakers** section lists one row per speaker with a rename box, mirroring
  the live Speakers legend. Type a name, press Enter, and the transcript
  above repaints with it. It appears only for a recording whose meeting
  folder still holds `meeting.json`, and it updates that same name map plus
  the stored, searchable transcript text. Note that this after-the-fact
  rename rewrites the Library item's stored transcript, so it **refuses** —
  with the notice "This transcript came from ingest; rename the live
  transcript in Meetings.", changing nothing on disk or in the database —
  whenever that stored text is not the meeting's own render, which is the
  case whenever the offline ingest pass produced the Library copy
  (`post_transcribe` left on). It **does** work on either shape the app
  itself writes: the plain `[hh:mm:ss] Name: text` transcript, and the
  Markdown `transcript.md` that goes to the Library when `post_transcribe`
  is off — a rename re-renders in whichever shape the item already has, so
  a Markdown transcript keeps its header and its `**Name:**` lines. It
  refuses the same way when the recording folder's `transcript.jsonl` is
  missing or empty. When it does go through, the replaced text is kept as a
  document version, so the change can be rolled back. If the write fails —
  a concurrent edit to the same item, say — nothing is left half-applied:
  the name map on disk is put back, and the rename can simply be retried.
  Whatever you type is stored as typed apart from control characters, which
  are dropped; a name containing Markdown punctuation (`*`, `` ` ``, `[…]`)
  shows as those characters in a Markdown transcript rather than turning
  into formatting or a link.
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
- **Pause/Resume, the device pickers' remembered choice, and the level
  meters are covered by automated pilot tests, not by this page's live
  session** — the mic level stayed at 0% throughout live verification (no
  audio input reached the terminal running the app), so those three
  behaviors were exercised only in the test suite, not watched moving on a
  real screen.

—
*Verified against dev @ 15254e860 + feat/meeting-onnx-diarizer @ 3b82d4763 —
2026-09-07. That branch added the torch-free ONNX (sherpa-onnx) live diarizer
(TASK-31827): the "Speaker-label engines and their models" section and the
three new `[meetings]` keys (`diarizer_backend`'s four values, `onnx_embedder`,
`onnx_models_dir`) were verified by reading `tldw_chatbook/Audio/meeting_owner.py`
(`DIARIZER_BACKENDS`, `ENGINE_MODULES`, `AUTO_ORDER`, the three field
validators), `tldw_chatbook/Audio/diarizer_engine_onnx.py` (the manifest's file
names, sizes, hashes and licences, the download allowlist and 10-minute budget,
`onnx_models_dir` semantics) and
`tldw_chatbook/UI/Screens/meetings_screen.py` (`_live_diarization_copy`,
`_tick_diarizer_status`, `_onnx_download_mb`; the final fix wave re-read the
last of those for the split "off (models unavailable)" / "off (backend
unavailable)" rows) — not by a live session: no
meeting has been recorded on this host with `live_diarization = true`, and no
model download has been watched on the rail. The engine-comparison paragraph
and the `voice_match_threshold` starting values are quoted from the measured
bake-off report in `Docs/STT_Evaluation/task-31827/`, whose numbers come from
one Apple M-series machine (the x86 runner half was never dispatched). Earlier
stamp: feat/meeting-voiceprint @ 74b771396 (its
final fix wave plus the Qodo review round, which added the transfer-path and
import-record refusals documented above) —
2026-09-07. That branch added self-voiceprint enrollment and matching
(TASK-31826): the "Remember my voice" section and the four `voice_*`
config keys above document `Enroll my voice`, the "Voice match: …" rail
line, the matched-speaker marker and override, the post-Stop learning
offer, Delete/Export/Import, the keyring/key-file encryption modes, and
the privacy and threshold notes — verified by reading
`tldw_chatbook/Audio/meeting_owner.py` and
`tldw_chatbook/UI/Screens/meetings_screen.py` directly (copy strings,
widget ids, and config defaults), and by the pilot/unit coverage in
`Tests/UI/test_meetings_screen.py` and `Tests/Audio/test_voiceprint_store.py`,
not by a live session — this host has never granted the app a real
microphone enrollment. Two things this section says honestly rather than
aspirationally: there is no Settings › Meetings page yet (every control is
on the Meetings rail), and the "best similarity" calibration number is not
shown on screen (only the opt-in `Tests/Audio/test_voiceprint_real.py`
prints one, on a real Mac with `say`). The final fix wave corrected five
statements on this page that did not match the code — the learning offer's
trigger and its enrollment precondition, the "Voice match:" line's
position, the Delete/Export/Import row, Merge's model check (enforced on
accept, not by hiding the button), and what opening the screen does to the
key — and added the `live speaker labels off` reason, which is what the
SHIPPED default (`live_diarization = false`) now reports instead of "Voice
match: on". Earlier stamp: feat/meeting-followups @ 97a823256 + the PR #2471 Qodo fix
wave — 2026-09-06. That wave added the three sentences above about the
start-stamped display name, the retryable failed rename, and control
characters / Markdown punctuation in a typed name; all three are covered by
pilot/unit tests
(`Tests/UI/test_meetings_screen.py::test_a_display_name_change_mid_meeting_never_splits_the_live_rows`,
`Tests/UI/test_library_media_speaker_rename.py::test_a_failed_rename_restores_meeting_json_so_a_retry_can_succeed`
and its Markdown twin, plus the two name-sanitization cases in the same
file), not by a live session. Earlier stamp: feat/meeting-followups @
e060f8d27 + the final-review fix wave — 2026-09-06. That wave made the "either shape the app itself writes"
sentence above true (the rename used to refuse every `post_transcribe =
false` recording, which this page had implied it accepted); it is covered by
`Tests/UI/test_library_media_speaker_rename.py::
test_rename_works_on_the_markdown_transcript_and_keeps_its_shape`, not by a
live session. Earlier stamp: feat/meeting-diarization @ 14bbf2a7f + the PR
#2456 fix wave — 2026-09-06. The rows added in that wave (the "Live speaker labels"
status line, the footer's "Speaker labels unavailable" / "Speaker merge to
resolve" copy, and the Library rename's refusal rules) are covered by pilot
tests in `Tests/UI/test_meetings_screen.py` and
`Tests/UI/test_library_media_speaker_rename.py`; they were NOT re-verified
in a live session (this host still has no System Audio Recording grant).
Earlier stamp: e26193495 — 2026-09-05.
The "System source lost" rail copy above was verified by pilot test
(`Tests/UI/test_meetings_screen.py::test_lost_tap_updates_system_status`),
not in a live session — this host has no System Audio Recording grant, so
no real tap loss was observed. Everything else on this page carries the
2026-09-04 live verification described in the Quirks section. The Speakers
legend and rename box (task 7) and the after-the-fact Library rename (task
8) are covered only by pilot/unit tests —
`test_legend_row_mounts_and_rename_input_updates_ui` and the `_apply_rename`
unit cases in `Tests/UI/test_meetings_screen.py`, plus
`Tests/UI/test_library_media_speaker_rename.py` — not by a live session with
`live_diarization` turned on. The Library reader's "Rename speakers" section
(TASK-31745) closed the reachability gap this page used to document, and is
itself covered only by pilot tests
(`Tests/UI/test_library_media_viewer_speaker_rename.py`): the section, one
submitted rename, and the refusal notice were exercised in the test suite,
not watched in a running app.*
