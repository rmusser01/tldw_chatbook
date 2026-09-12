# Console: voice & hands-free — speaking to the Console and hearing replies

The Console has two voice features that work together:

- **Speak replies** (switch in the status row): new assistant replies in
  this conversation are spoken automatically. Per-conversation and
  consent-gated — the first flip asks you to confirm the TTS provider and
  destination, and changing that destination later asks again.
- **Hands-free** (switch beside it, or **Ctrl+Shift+H**): the full voice
  loop — speak, it sends, the reply is spoken, speak again.

While hands-free is running, it owns reply speech; the two features never
double-speak. **Speak replies** persists per conversation for normal
(keyboard) use.

## Hands-free: the loop

| Phase | What you see | What is happening |
|---|---|---|
| Listening | `● 0:07` + live recognized words | The mic is open; speech is recognized as you pause |
| Countdown | "hands-free · sending in 1.5s…" | A pause finalized a segment; it sends when the countdown ends unless you keep talking |
| Thinking | "hands-free · thinking…" | Your message was sent; the reply is being generated |
| Speaking | "hands-free · speaking" | The reply streams out sentence by sentence |

- **Interrupt a spoken reply** by typing anywhere in the composer, or press
  **Esc** / **Ctrl+Shift+H** to exit the loop from any phase.
- The mic closes while a reply is generated/spoken (no echo cancellation),
  then reopens for your next turn.
- A capture ends itself at 60 seconds — in the normal pipeline that ends
  the turn (retained text lands in your draft); it is not a turn boundary.

### Spoken commands

Whole-phrase only, after the wake word (default "Console"):

| Say | Effect |
|---|---|
| "Console, send." | Send the dictated text immediately (skips the countdown) |
| "Console, stop." | Stop the capture / exit the loop |
| "Console, discard." | Throw away the dictated text |
| "Console, new paragraph." / "new line" | Insert a line break |
| "Console, read that back." | Speak the dictated text |
| "Console, new session." | Start a new conversation |
| "Console, hands free." | Enter the loop (mic button / switch / Ctrl+Shift+H also work) |

Anything else is dictated text — an ambiguous phrase is never treated as a
command. Change the wake word with `dictation.command_prefix` in
config.toml.

### Tuning

Settings ▸ Speech & TTS ▸ **Realtime engine** section (works for the
pipeline engine too):

- **Send delay (seconds)** — how long the countdown runs after you pause
  (default 1.5). Blank keeps the default.
- **Acoustic barge-in (headphones)** — lets your *voice* interrupt a
  speaking reply. Off by default: there is no echo cancellation, so on
  speakers the recognizer hears the reply itself. Use headphones.
- **Hands-free engine** — Auto / Pipeline / Realtime (the optional
  low-latency provider-hosted engine; see
  [Settings](../settings.md) for its privacy/cost notes).

## Speak replies: consent and failures

The first time you enable it, a dialog names the TTS provider and
destination and whether charges may apply. If you later switch TTS
provider, endpoint, or voice profile destination, consent is asked again
("The speech destination changed. Confirm Speak replies again.") — that is
the destination-fingerprint guard, not a glitch.

If speaking fails, automatic speech pauses itself and two buttons appear:
**Retry speech** (retry the reply that failed) and **Resume auto-speak**
(keep speaking future replies). A hands-free reply in which *every*
sentence fails produces one error notice with remedies.

## Setup by platform

Voice needs three local pieces: **microphone capture**, **speech-to-text**,
and **audio output**. The app checks them when hands-free starts and tells
you exactly what is missing.

| Piece | Install | Notes |
|---|---|---|
| Mic capture + VAD | `pip install 'tldw_chatbook[speech_recording]'` | Includes `sounddevice` (mic + playback) and `webrtcvad`. On Linux, building `pyaudio` (also included) needs `dnf install portaudio-devel gcc` — `sounddevice` alone is enough to capture |
| Speech-to-text | `pip install 'tldw_chatbook[transcription_faster_whisper]'` | First run downloads the model (several minutes) |
| Reply audio output | Built in for WAV/PCM via `sounddevice`; compressed formats (MP3/Opus/AAC/FLAC) need a player binary | Linux: `dnf install mpv` (or `apt install mpv`). `pw-play`/`paplay` cover WAV/FLAC |

If a machine cannot play the configured format at all, Console speech
automatically requests **WAV** instead — you can also set Settings ▸
Speech & TTS ▸ Output format to WAV to make it explicit. When nothing can
play audio, hands-free warns once on entry ("replies will be silent") and
each fully-silent reply says so.

### Linux notes (Fedora and friends)

- Stock Fedora ships **no** player binary the legacy path can use (`mpv`,
  `mplayer`, `ffplay`, `aplay`, `paplay` are absent; `pw-play` covers
  WAV/FLAC only). Install `mpv`, or use WAV output — otherwise reply
  speech depends on the automatic WAV fallback.
- macOS needs nothing extra (`afplay` is built in).
- Without `webrtcvad` the loop runs **degraded**: nothing is auto-sent
  (retained speech lands in your draft), and spoken commands cannot fire
  mid-capture — the entry warning says this. Esc, the mic button, or
  Ctrl+Shift+H still end a capture.

## Troubleshooting

- **"Microphone support isn't installed…"** — install the
  `speech_recording` extra; hands-free refuses to start until then.
- **Switch flips but nothing happens / replies are silent** — check the
  toast: either a player is missing (install `mpv` or set Output format to
  WAV) or the TTS provider failed (Settings ▸ Speech & TTS).
- **"No audio was captured from the microphone."** — check the OS
  microphone permission for your terminal.
- **Turns send too early / too late** — adjust **Send delay** in Settings
  (or `dictation.handsfree_send_delay_seconds`).

## Related docs

- [Attachments, images & voice](attachments-images-voice.md) — one-shot
  dictation into the composer (the Mic button).
- [Speech services](../../Features/Speech-Services-Guide.md) — STT/TTS
  provider deep dive.
- [Settings](../settings.md) — Realtime engine section and global speech
  defaults.
