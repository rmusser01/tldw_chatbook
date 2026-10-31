# Console: voice & hands-free — speaking to the Console and hearing replies

The Console has two voice features that work together:

- **Speak replies** (in the status row, reads **Speak replies: Off**, **On**
  or **Paused**): new assistant replies in this conversation are spoken
  automatically. Per-conversation and consent-gated — turning it on asks
  you to confirm the TTS provider and destination, and changing that
  destination later asks again.
- **Hands-free** (beside it, reads **Hands-free: Off**, **Starting…** or
  **On**; or **Ctrl+Shift+H**): the full voice loop — speak, it sends, the
  reply is spoken, speak again. Turning it on asks first (see below).

Each control's word is its state. Tab to one and press **Enter** or
**Space** to change it; a focused control shows its word as a solid bar, so
the value stays readable while it has focus.

While hands-free is running, it owns reply speech; the two features never
double-speak. **Speak replies** persists per conversation for normal
(keyboard) use.

## Living character portraits

Prepare a character's six mouth/blink poses in [Personas → Living portrait](../roleplay-chat-dictionaries/characters-and-personas.md).
The Console's Character section then follows that character's speech, with mouth
movement reacting to measured audio and closing during pauses. Blinking and gentle
idle movement are independent. **Static**, **Reduce motion**, animation/reaction
settings and hiding the portrait retain their existing behavior.

The radio status reads **Ready**, **Preparing voice**, **Transmitting**, **Paused** or
**Voice unavailable**. The level bar requires fresh qualified output timing; silence
empties it. An external file player can report actual playback status while leaving
the mouth closed and bar hidden. It does not estimate lip sync from text or elapsed
file duration. Changing conversations or profiles prevents an old speech owner from
animating the new character. Normal speech transitions keep sidebar rows stable;
compact layouts omit the level bar first.

Eligible PCM streaming and complete PCM16 WAV artifacts use local device timing.
Natural completion waits for the device's finished acknowledgement and any validated
pending output tail before the next sentence or listening handoff. **Stop** still interrupts immediately. Synchronization
accuracy depends on the selected player/device; the source qualification record
lists exercised paths and measurement limits.

Pocket TTS can use the existing [OpenAI-compatible TTS settings](../openai-compatible-tts.md)
with its native `/tts` URL and WAV output. This endpoint's WAV must finish arriving
before playback starts; once playing through the local PCM16 WAV path, the portrait
uses measured device timing.

## How formatted replies are spoken

**Speak replies** and a completed message's **Speak** action turn Markdown
into spoken text:

- Headings, emphasis, lists, and quotations keep their words, with pauses
  between blocks. Links use their labels; images use their alternative text.
- Checklist items announce **“Checked”** or **“Unchecked”** before their words.
- HTML formatting keeps visible words and pauses at line breaks and block
  boundaries. Comments and script/style bodies are skipped.
- Inline code is read without backticks, including commands and identifiers.
- Each fenced or indented code block is replaced by **“Code block omitted.”**
  The code remains available in the onscreen response.
- Tables are read row by row with column labels, such as “Fruit: Apples;
  Count: 2.”

Retry and global voice fallback use the same conversion. Replies containing
only formatting are skipped without pausing automatic speech. Both the original
reply and the spoken version must fit the existing 5,000-character speech limit.

## Hands-free: the loop

| Phase | What you see | What is happening |
|---|---|---|
| Listening | `● 0:07` + live recognized words | The mic is open; speech is recognized as you pause |
| Countdown | "hands-free · sending in 1.5s…" | A pause finalized a segment; it sends when the countdown ends unless you keep talking |
| Thinking | "hands-free · thinking…" | Your message was sent; the reply is being generated |
| Speaking | "hands-free · speaking" | The reply streams out sentence by sentence |

- **Turning it on asks first.** Hands-free opens your microphone and
  speaks every reply through the same TTS provider as Speak replies, so it
  shows the same dialog (provider, destination, whether charges may apply)
  before anything starts. **Cancel** or **Esc** leaves it Off with nothing
  running. With no TTS destination available, the dialog says replies will
  not be spoken.
- **Starting…** means the speech model is still loading (minutes on a
  first run); the mic button reads **Cancel** meanwhile. Leaving Hands-free
  then (**Esc**, the control, Ctrl+Shift+H or **Cancel**) cancels it:
  nothing is recorded. The load itself runs to the end in the background,
  and the capture it would have opened is discarded.
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

If speaking fails, automatic speech pauses itself — the control reads
**Speak replies: Paused** — and two buttons appear: **Retry speech** (retry
the reply that failed) and **Resume auto-speak** (keep speaking future
replies). A hands-free reply in which *every*
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
- **Control turns on but nothing happens / replies are silent** — check the
  toast: either a player is missing (install `mpv` or set Output format to
  WAV) or the TTS provider failed (Settings ▸ Speech & TTS).
- **"No audio was captured from the microphone."** — check the OS
  microphone permission for your terminal.
- **"Dictation failed: The '…' speech model could not be downloaded or
  loaded."** — nothing was recorded and Hands-free turns Off. Check the
  network and turn Hands-free on (or press **Dictate**) to try again, or
  choose another model in Settings ▸ Speech & TTS; the app log names the
  underlying error. "…showed no download progress for 120 seconds" means the
  load went two minutes without a byte landing in the Hugging Face cache;
  raise `dictation.model_prepare_stall_seconds` if your model needs longer.
- **Turns send too early / too late** — adjust **Send delay** in Settings
  (or `dictation.handsfree_send_delay_seconds`).

## Related docs

- [Attachments, images & voice](attachments-images-voice.md) — one-shot
  dictation into the composer (the Mic button).
- [Speech services](../../Features/Speech-Services-Guide.md) — STT/TTS
  provider deep dive.
- [Settings](../settings.md) — Realtime engine section and global speech
  defaults.
