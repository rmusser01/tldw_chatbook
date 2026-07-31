# Speech Services User Guide

## Overview

The Speech Services feature in tldw_chatbook provides comprehensive Text-to-Speech (TTS) and Speech-to-Text (Dictation) capabilities with a focus on privacy, usability, and integration.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Text-to-Speech (TTS)](#text-to-speech-tts)
3. [Dictation (Speech-to-Text)](#dictation-speech-to-text)
4. [Voice Commands](#voice-commands)
5. [Privacy & Security](#privacy-security)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Features](#advanced-features)

## Getting Started

### Accessing Speech Services

1. Navigate to the **Speech Services** tab (formerly STTS)
2. You'll see two main sections:
   - **Text-to-Speech (TTS)**: Convert text to natural speech
   - **Dictation (STT)**: Convert speech to text

### Quick Start

#### For TTS:
1. Click "🎮 Playground" in the TTS section
2. Enter text in the input area
3. Select voice and settings
4. Click "Generate" to create audio

#### For Dictation:
1. Click "🎤 Live Dictation" in the Dictation section
2. Click "Start Dictation" or press `Ctrl+D`
3. Speak clearly into your microphone
4. Click "Stop" when finished

## Text-to-Speech (TTS)

### TTS Playground

The TTS Playground allows you to experiment with different voices and settings:

- **Text Input**: Enter or paste the text you want to convert
- **Provider Selection**: Choose audio.cpp, OpenAI, ElevenLabs, Kokoro (local),
  or another configured provider
- **Voice Selection**: Pick from available voices for the selected provider
- **Format**: Choose output format (MP3, WAV, etc.)
- **Provider Settings**: Adjust provider-specific parameters

### TTS Settings

Configure default TTS options:

- Default provider, model mode, and voice mode
- Audio format preferences
- API key management
- Voice blend creation (Kokoro only)

### Voice generation profiles

Chatbook can save and manage reusable exact audio.cpp generation selections.
The local profile database defaults to `tldw_chatbook_tts_profiles.db` in the
current Chatbook user data directory. Advanced installations may override it
in the configuration file:

```toml
[database]
tts_profiles_db_path = "/absolute/path/to/tts-profiles.db"
```

To create and use a profile:

1. Connect to an existing audio.cpp server and generate a complete WAV in the
   **TTS Playground**.
2. Play the result to confirm it is the voice you want.
3. Choose **Save result as profile** and enter a unique name. Chatbook saves
   the provider, model, optional exact voice, WAV format, and speed from that
   successful result—not from selectors changed afterward.
4. Open **Voice profiles** to search, page through, or refresh the library.
5. Select a row to **Preview**, **Edit**, **Duplicate**, or **Delete** it.
   Preview opens the Playground with the saved exact values without generating
   speech; choose **Generate Speech** there when ready.

Availability is observational and never rewrites a saved profile. **Available**
means the current external server advertises the exact saved selection.
**Unavailable** means the current authoritative catalog does not support it;
refresh capabilities or edit the profile. **Unverified** means Chatbook could
not make an authoritative determination, usually because discovery failed or
became stale; refresh and retry.

Profile-store failures are isolated from ordinary speech. If the library
reports that profile storage is unavailable, the Playground and Console can
still generate speech; choose **Refresh** in the library to retry opening or
loading the store.

Character authority acquisition, character assignment, roleplay voice routing,
legacy-provider profile execution, and profile/card portability or
synchronization remain deferred. This slice also does not manage an audio.cpp
server process.

Profiles are owned locally and contain generation choices, not provider
connection details. They exclude server origins, credentials and API keys,
binary or `server.json` paths, managed-process settings, message text, and
provider health observations. Names compare uniquely after Unicode
normalization and case folding. A saved edit must use the revision originally
loaded, so a concurrent change is reported as a conflict instead of being
silently overwritten.

**Database Tools → Backup All Databases** includes the profile store when its
repository is available. Chatbook creates that entry with SQLite's online
backup mechanism rather than copying an open database file. Each database
backup is internally consistent, but the collection is not a single
cross-database atomic snapshot. If the profile backup cannot be produced, the
operation reports a partial failure instead of claiming that profiles were
backed up.

Profile restore is an explicit, bounded repository operation; there is not yet
a profile-specific restore control in Speech Services. The repository validates
the candidate and creates a recovery copy before replacement. Failures before
replacement preserve the current store. If replacement occurs but the new
store cannot be safely reopened, profile storage remains unavailable with
recovery evidence rather than silently creating an empty database. Do not
replace an open profile database with a raw file copy.

See
[ADR-028](../../backlog/decisions/028-character-tts-generation-profile-ownership.md)
for the ownership, privacy, backup, and deferred-scope decisions.

### Using an Existing audio.cpp Server

Chatbook can connect the Playground to one compatible `audiocpp_server` that
you start and manage yourself:

1. Start your compatible `audiocpp_server` and note its HTTP or HTTPS origin.
2. Open **Speech Services → TTS Settings → audio.cpp External Server**.
3. Enter the server's **Base URL** and review the timeout and safety bounds.
4. Choose audio.cpp as the default provider. Select **First available** or an
   exact model, then choose **Server default** or an exact voice.
5. Click **Save Settings** once. Saving validates and persists the values but
   does not connect to or launch a server. The new defaults are available to
   Console **Speak** without restarting Chatbook.
6. Click **Test Connection** or **Refresh Models**. These actions use the saved
   settings and discover the server's TTS models.
7. Return to **Playground**, select **audio.cpp**, then choose a discovered
   model. Voice discovery happens when the model is selected.
8. Leave **Server default** selected to omit the voice from the request, or
   select an announced voice.
9. Enter text and click **Generate Speech**. After the complete WAV is
   validated, use **Play** or **Export**.
10. In Console, use **Speak** on a response to synthesize and play it with the
    same saved defaults.

Console offers **Speak** only on completed assistant responses. When selected,
Chatbook captures the exact visible response and selected variant in a
temporary immutable snapshot, then checks that the same response, conversation
branch, durable message version when present, and assistant identity are still
current before starting TTS. If the response changed, Chatbook asks you to
select **Speak** again instead of speaking stale text. The snapshot is never
saved and does not select a character voice profile; Console continues to use
the global defaults configured above.

audio.cpp currently returns one complete WAV per request and supports speed
`1.0`; the Playground locks both controls while it is selected. This still uses
the asynchronous adapter interface, but it is not incremental audio streaming.
Switching to another provider restores that provider's previous model, voice,
format, speed, and provider-specific controls.

If discovery becomes stale or fails, the prior choices may remain visible but
new generation stays disabled until a successful retry or model refresh.
Already generated audio remains playable and exportable. If a selected model or
voice disappears after refresh, the Playground announces and selects a valid
fallback. A failed audio.cpp request never silently uses another model or
provider.

Older audio.cpp settings with blank model or voice values are interpreted as
**First available** and **Server default** when their mode keys are absent.
Saving writes explicit mode keys. Exact selections retain compatibility aliases
for older Chatbook builds; dynamic selections remove stale exact values. Before
downgrading while using a dynamic selection, save explicit model and voice
values or restore a trusted pre-feature configuration backup.

If a save reports **Saved — applying after current speech**, the current
admitted response is allowed to finish. Chatbook applies only the latest saved
generation, without running old and replacement audio.cpp adapters together.

**Privacy:** the text submitted for synthesis is sent to the configured
audio.cpp server. Chatbook avoids putting that text, the configured origin or
settings values, credentials, raw remote responses, and unsafe remote
identifiers in normal UI diagnostics or application logs.

This release does not download, launch, monitor, restart, or stop audio.cpp and
does not accept a binary path or `server.json` path. User-provided binary plus
user-provided `server.json` launch and supervision are deferred to later
managed-mode slices.

Release validation used an isolated clean configuration and a user-owned
audio.cpp server at `127.0.0.1:8080`. A deterministic Mira Console response
produced one complete owner-only WAV, played once through `/usr/bin/afplay`, and
left the same external listener healthy after Chatbook shut down. A later
post-rebase live rerun was not attempted because no server was listening;
Chatbook did not start the installed binary.

### AudioBook Generator

Create long-form audio content:

1. Import content from:
   - Text files
   - Notes
   - Conversations
   - Clipboard
2. Configure chapter detection
3. Select narrator voice
4. Generate complete audiobook

### Voice Cloning

Create custom voices (ElevenLabs only):

1. Click "🎭 Voice Cloning"
2. Upload voice samples
3. Configure voice settings
4. Use cloned voice in any TTS generation

## Dictation (Speech-to-Text)

### Live Dictation Interface

The improved dictation interface offers:

- **Real-time Transcription**: See text as you speak
- **Privacy Controls**: Choose local-only processing
- **Voice Commands**: Control with voice
- **Troubleshooting**: Built-in audio diagnostics

### Privacy Settings

Control how your voice data is handled:

#### Save History
- **Off** (default): No transcriptions are saved
- **On**: Save transcriptions for later reference

#### Local Only Mode
- **On** (default): All processing happens on your device
- **Off**: May use cloud services for better accuracy

#### Auto-clear Buffer
- **On** (default): Audio data is deleted immediately after processing
- **Off**: Audio may be temporarily cached

### Using Dictation

1. **Start Dictation**:
   - Click "🎤 Start Dictation" or press `Ctrl+D`
   - Grant microphone permissions if prompted

2. **While Dictating**:
   - Speak naturally at a normal pace
   - Pause briefly between sentences
   - Use voice commands for formatting

3. **Pause/Resume**:
   - Click "⏸️ Pause" or press `Ctrl+P`
   - Useful for thinking or interruptions

4. **Stop Dictation**:
   - Click "🛑 Stop" or say "stop dictation"
   - Review and edit the transcript

### Integration with Other Features

#### Send to Chat
1. After dictating, click "Send to Chat"
2. Text appears in active chat conversation

#### Send to Notes
1. Click "Send to Notes"
2. Create new note or append to existing

#### Voice Input Button
- Look for the "🎤 Voice" button next to text fields
- Click to add voice input anywhere in the app

### Console Voice Commands (V2)

The Console's dictation capture (the composer's mic button) recognizes a
small set of spoken commands while the microphone is open, so a capture can
be controlled — and, with spoken feedback on, operated — without touching
the keyboard, the mouse, or the screen. This is separate from the generic
"Voice Commands" list below, which applies to the standalone Dictation
window.

**Activation.** Say the configured prefix word (`dictation.command_prefix`,
default `"console"`) immediately followed by one of the command phrases
below, as the *entire* spoken segment — nothing before it, nothing after.
For example: "Console, stop." A segment that only starts with the prefix but
goes on to say other things — "console send button is broken" — is not a
command; it fails the match and lands in the transcript as ordinary text.

**Fail-open, not fail-silent.** Normalization lowercases the segment,
strips all punctuation (leading, trailing, and internal), and collapses
whitespace, then checks the result against `<prefix> <phrase>`. Anything
that isn't an exact whole-segment match — plain dictation, a misheard
command like "console sned" — is emitted as ordinary text into the draft,
never dropped and never actioned silently. There is no error for an
unrecognized command; the words simply appear where you can read, edit, or
delete them.

**Command table:**

| Say (after the prefix) | Kind | Effect |
|---|---|---|
| "new paragraph" | inline | Inserts a paragraph break (`\n\n`); capture keeps running. |
| "new line" | inline | Inserts a line break (`\n`); capture keeps running. |
| "stop" | capture-ending | Ends the capture and inserts the accumulated text at the caret — the same as pressing the mic button again. |
| "send" | capture-ending | Ends the capture, inserts the text, and sends the message once insertion has completed. Refused if you switched tabs while it was transcribing (the text is in the *original* tab's draft, so sending would ship the other tab's), or if Send is blocked for any other reason — the refusal says which. |
| "discard" | capture-ending | Ends the capture without inserting anything — the same as pressing Cancel. No confirmation is asked; saying it is treated as explicit intent. Once the capture has moved on to transcribing there is nothing left to abort, so it is refused with "Too late to discard" and the text still lands. |
| "read that back" | capture-ending | Ends the capture (inserting the text first), then speaks the latest **completed** assistant reply. If the reply is still streaming, or there is none yet, it acknowledges instead of speaking a partial answer. |
| "new session" | capture-ending | Ends the capture (inserting the text first), then opens a new session tab. |

Inline commands leave the capture open; every other command in the table
ends it. A capture whose spoken segments were entirely commands — for
example, dictating nothing but "console, stop" — is not treated as a failed
or empty capture: it produces no error and inserts no stray text, only a
"Nothing to insert." notice so a capture that dropped your break is not
silently indistinguishable from one that landed.

Every refusal says so rather than looking like a missed command.
"Session changed — not sent.", "Too late to discard — text inserted." and
"Nothing to insert." each show as a notice always, and are spoken as well
when `spoken_feedback` is on. A `send` that Send itself refused (empty draft,
a run already in flight, send blocked, a typed `/`-command) already shows
that reason in its own notice, so the spoken ack is just "Not sent." — never
"Sent." for a message that never went out.

**Choreography and latency.** A command only fires once its segment
finalizes, and a segment finalizes on a pause in speech — so **pause
briefly both before and after speaking a command**. Concretely: the
recorder runs voice-activity detection over 20 ms frames and only forwards
the ones that contain speech; the dictation service accumulates whatever it
is forwarded and, once the recorder has forwarded nothing for the whole
`dictation.silence_threshold_seconds` (default 2.0 s), transcribes that
*entire* accumulated segment in one call and finalizes it. There is no
periodic transcription cycle any more — a segment's audio is only ever sent
to the speech model once, at that pause (or at stop). So the realistic
budget for a command to fire is `silence_threshold_seconds` **plus one
whole-segment transcription** — measured 4-5 s for a short utterance on a
loaded machine with a local Whisper-family model; faster on an idle machine,
and it scales with how much was said, not with a fixed buffer interval. This
is expected behavior, not a bug, and lowering the threshold shortens the
pause half of it (at the cost of finalizing plain dictation into more,
shorter segments) — it does not shorten the transcription itself. Because
the pause *before* the command and the pause *after* it must each complete
their own threshold-plus-transcription cycle before their segments
finalize, a full "pause, command, pause" round trip costs roughly **two**
threshold intervals **plus two whole-segment transcriptions**, not one of
each — budget for that rather than reading it as lag.

Voice-activity detection needs the optional `webrtcvad-wheels` package
(installed by the `speech_recording` extra; it still imports as `webrtcvad`,
unchanged). Without it the recorder forwards every frame, nothing ever looks
like a pause, and segments finalize only when the capture stops — so inline
commands and mid-capture finalization do not fire, though dictation itself
still works end to end: the *entire* capture accumulates as one segment and
is transcribed in a single call once you stop, bounded by
`dictation.stop_join_timeout_seconds` (default 30 s) rather than by
anything mid-capture. The Console notifies once per run when this happens.

**Configuring the prefix.** `dictation.command_prefix` accepts multi-word
prefixes and is normalized the same way as spoken commands. Leaving it
blank (or whitespace-only) falls back to the default `"console"` rather
than matching every segment as prefixed.

**A known, accepted false-fire.** Because punctuation is stripped before
matching, staccato dictated prose that finalizes as one segment can
normalize to the same text as a real command and fire it — the canonical
example is "Console. Send." finalizing as a single segment and firing
`send`. This trade-off is deliberate: keeping punctuation would let the
comma that recognizers almost universally insert after a vocative prefix
("Console, send.") break every real command, i.e. the feature would never
fire at all. The false-fire is rare and visible — an inline break acknowledges itself in
the composer's voice chip as `¶`, and a capture-ending command's evidence is
the action itself: the chip flips to `◌ Transcribing…`, the capture visibly
ends, and the command's effect (send, discard, new tab, read-back) follows
on the spot — and it is one of the checks in live verification before every
release.

**Spoken feedback (opt-in).** `dictation.spoken_feedback` (default
`false`) — when on, the Console speaks capture-ending acknowledgements
("Sent.", "Discarded.", "New session.") and dictation error messages,
through the same speech pipeline as per-message Speak. **"Capture started"
is deliberately never spoken:** the microphone is already open by the time
a capture starts, so speaking it would talk over the open mic and get
transcribed straight back into the draft. With the toggle off, none of this
speaks; "read that back" is the one exception and always speaks, since it's
an explicit request rather than ambient feedback.

Status speech never overlaps an open microphone: inline commands only
acknowledge via the voice chip (never spoken), and capture-ending commands
speak only after the capture has fully closed. **Starting a new capture
always stops any speech that is currently playing first** — a status ack or
an in-flight "read that back" — because the single-slot audio player only
stops a clip when a *new* one starts, and opening the microphone plays
nothing on its own. Without this rule, playback still running at capture
start would be picked up by the open mic and transcribed into the new
draft.

## Voice Commands

### Built-in Commands

#### Text Manipulation
- "new paragraph" - Insert paragraph break
- "new line" - Insert line break
- "delete last word" - Remove the last word
- "delete last sentence" - Remove the last sentence
- "clear all" - Clear all text
- "undo that" - Undo last action

#### Punctuation
- "comma" - Insert ,
- "period" - Insert .
- "question mark" - Insert ?
- "exclamation mark" - Insert !
- "semicolon" - Insert ;
- "colon" - Insert :
- "open quote" / "close quote" - Insert "

#### Formatting
- "make bold" - Bold selected text
- "make italic" - Italicize selected text
- "capitalize that" - Capitalize selection
- "uppercase that" - Convert to UPPERCASE
- "lowercase that" - Convert to lowercase

#### App Control
- "stop dictation" - Stop recording
- "pause dictation" - Pause recording
- "switch to chat" - Go to chat tab
- "switch to notes" - Go to notes tab
- "save this" - Save current content
- "show help" - Display help

### Custom Commands

Create your own voice commands:

1. Open dictation settings
2. Click "Voice Commands"
3. Click "Create Custom Command"
4. Define:
   - **Phrase**: What you'll say
   - **Action**: What happens
   - **Text**: Text to insert (optional)
   - **Description**: Help text

Example custom commands:
- "add bullet point" → Insert "• "
- "insert date" → Insert today's date
- "my signature" → Insert your email signature

## Privacy & Security

### Privacy-First Design

- **Default Settings**: Privacy mode enabled by default
- **Explicit Service Boundary**: Cloud providers and an externally configured
  audio.cpp server receive the content submitted to them
- **Local Processing**: Prefer on-device transcription
- **Encrypted History**: Optional history uses encryption

### Data Handling

#### What's Stored
- Transcriptions (only if history enabled)
- Your privacy preferences
- Custom voice commands

#### What's NOT Stored
- Audio recordings (deleted immediately)
- Voice biometrics
- Cloud service credentials in plain text

### Security Features

- Encrypted configuration files
- Secure API key storage
- Automatic data cleanup
- Permission-based access

## Troubleshooting

### Audio Device Issues

Use the built-in troubleshooter:

1. Click "🔧 Troubleshoot" in dictation window
2. The tool will:
   - Detect available microphones
   - Show input levels
   - Test recording
   - Provide specific solutions

### Common Problems

#### No Microphone Detected
- Check physical connection
- Grant app permissions:
  - **macOS**: System Preferences → Security & Privacy → Microphone
  - **Windows**: Settings → Privacy → Microphone
  - **Linux**: Check PulseAudio/ALSA settings

#### Poor Recognition Accuracy
- Speak clearly and at normal pace
- Reduce background noise
- Adjust buffer duration (Settings → Performance)
- Try a different provider

#### Dictation Won't Start
- Check error message for guidance
- Ensure microphone isn't used by another app
- Try troubleshooting tool
- Restart the application

### Performance Tuning

#### Buffer Duration
- **Lower (100-300ms)**: More responsive, may be less stable
- **Default (500ms)**: Balanced performance
- **Higher (1000-2000ms)**: More stable, slight delay

#### Provider Selection
- **Local providers**: Better privacy, work offline
- **Cloud providers**: Better accuracy, need internet

## Advanced Features

### Transcription History

If history is enabled:

1. Access via "📝 Transcription History"
2. Features:
   - Search past transcriptions
   - Filter by date/language
   - Export to various formats
   - Encrypted storage

### Voice Input Integration

Add voice input to any text field:

```python
# In your custom widget
from tldw_chatbook.Widgets.voice_input_button import VoiceInputButton

# Add to your compose method
yield VoiceInputButton(
    target_widget_id="my-input",
    on_result=self.handle_voice_input
)
```

### Batch Processing

For multiple transcriptions:

1. Queue multiple audio files
2. Select consistent settings
3. Process in background
4. Export results together

### API Usage

For developers extending the system:

```python
# Get dictation service
from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

service = LazyLiveDictationService(
    language='en',
    enable_punctuation=True
)

# Start dictation
service.start_dictation(
    on_partial_transcript=handle_partial,
    on_final_transcript=handle_final
)
```

## Tips & Best Practices

### For Best Recognition

1. **Environment**:
   - Quiet room with minimal echo
   - Consistent distance from microphone
   - Good quality headset or microphone

2. **Speaking Style**:
   - Natural pace (not too fast/slow)
   - Clear articulation
   - Pause between sentences
   - Avoid filler words

3. **Settings**:
   - Match language setting to your speech
   - Enable punctuation for cleaner output
   - Use appropriate provider for your needs

### Privacy Recommendations

1. **Maximum Privacy**:
   - Enable "Local Only" mode
   - Disable history saving
   - Use local TTS providers
   - Enable auto-clear buffer

2. **Balanced Approach**:
   - Use local providers when possible
   - Enable encrypted history
   - Review privacy settings regularly

3. **Convenience Focus**:
   - Enable all features
   - Use cloud providers for accuracy
   - Keep history for reference

## Keyboard Shortcuts

### Global
- `F1`: Show help
- `Ctrl+T`: Switch to TTS mode
- `Ctrl+D`: Switch to Dictation mode

### Dictation
- `Ctrl+D`: Start/Stop dictation
- `Ctrl+P`: Pause/Resume
- `Ctrl+C`: Copy transcript
- `Ctrl+E`: Export transcript
- `Ctrl+Shift+C`: Clear transcript

### TTS
- `Ctrl+G`: Generate speech
- `Ctrl+R`: Random example text
- `Ctrl+L`: Clear text
- `Ctrl+P`: Play audio
- `Ctrl+S`: Stop audio

## Configuration

### Config File Location
- `~/.config/tldw_cli/config.toml`

### Dictation Settings
```toml
[dictation]
provider = "auto"
language = "en"
punctuation = true
commands = true
buffer_duration_ms = 500
# How long a stop waits for the transcription thread to drain what is left of
# the capture. Must comfortably exceed one transcription: a warm local model
# takes about a second, a large one longer. If a stop reports "Transcription
# did not finish before dictation stopped", raise this (or pick a faster
# model). Invalid or non-positive values fall back to the 30s default.
stop_join_timeout_seconds = 30.0
# Load the speech model before the microphone opens, rather than lazily on the
# first audio chunk (which recorded into a void while the model downloaded).
# Set to false to skip it: the model then loads during the capture, as it used
# to. Default true.
warm_model_before_capture = true
# Prefix word(s) that activate a spoken command mid-capture in the Console,
# e.g. "console, stop". Normalized the same way as spoken segments (lowercase,
# all punctuation stripped, whitespace collapsed). A blank or whitespace-only
# value falls back to this default rather than treating every segment as
# prefixed. See "Console Voice Commands" below.
command_prefix = "console"
# Speak capture-ending acknowledgements ("Sent.", "Discarded.", "New
# session.") and dictation error messages aloud, through the same speech
# pipeline as per-message Speak. Off by default. "Capture started" is never
# spoken (the microphone is already open by then); "read that back" always
# speaks regardless of this setting, since it is an explicit request rather
# than ambient feedback.
spoken_feedback = false
# Pause length, in seconds, that finalizes a spoken dictation segment -- and
# therefore the delay before a spoken command executes. Must be a finite,
# positive number; invalid or non-positive values fall back to this 2.0s
# default. Lowering it shortens command latency at the cost of shorter,
# choppier dictation segments.
silence_threshold_seconds = 2.0
# How aggressively the recorder's VAD classifies a frame as speech, 0-3.
# Must be an integer in that range; invalid values fall back to this default.
# Lower values admit more ambient noise as speech and can prevent
# pause-finalization entirely.
vad_aggressiveness = 3
# Speech-to-text model dictation uses. Model resolution is PROVIDER-SCOPED
# and, when unset, never inherits `[transcription] default_model` -- that
# key is not scoped to any particular provider, so it may well name a model
# that belongs to a different provider than the one dictation resolved to
# (e.g. a faster-whisper model name handed to parakeet-mlx, which tries to
# load it as a HuggingFace repo and 404s -- this used to kill the capture
# outright). Concretely:
#   - Unset (the default) and the resolved provider is faster-whisper:
#     dictation picks "base" rather than inheriting `[transcription]
#     default_model` -- measured on real hardware, faster-whisper's own
#     default (distil-large-v3) took 11.5s to transcribe a short spoken
#     command under load, against 1.4s for "base"; a spoken command only
#     fires once its segment finalizes (see "Choreography and latency"
#     above), so that difference is the whole gap between commands feeling
#     instant and feeling dead.
#   - Unset and the resolved provider is anything else (parakeet-mlx,
#     parakeet-onnx, lightning-whisper-mlx, ...): dictation passes no model
#     at all, letting that provider's own transcription path load its own
#     default (parakeet-mlx: `mlx-community/parakeet-tdt-0.6b-v2`).
#   - Set (any provider): this value always wins, including setting it to
#     distil-large-v3 on purpose if accuracy matters more than latency.
# Blank/whitespace is treated as unset; a non-string value is ignored with a
# warning. The Console shows a one-time notice (once per app run) when the
# faster-whisper fast default actually displaces a differing
# `[transcription] default_model` you configured, so that specific case
# never happens silently.
#
# NOT a Console-only key: the standalone Dictation window (Speech >
# Dictation) already reads and writes this same `dictation.model` value, so
# a value set here also changes that window's model, and vice versa. That
# window has no control of its own for picking a model -- it only ever
# round-trips whatever this key already holds whenever some OTHER dictation
# setting there is changed -- so it cannot silently acquire a value on its
# own; the only way `dictation.model` gets set at all is by hand-editing
# config.toml, or by setting it here.
#
# First-run cost: if faster-whisper is your resolved provider and you have
# only a larger model (e.g. distil-large-v3) already downloaded, the fast
# default triggers a fresh "base" download on your first Console dictation
# capture. That warm-up (see "First run" below) is a single, non-fatal
# attempt before the microphone opens -- a failure there does not block the
# capture, but does NOT make the model load succeed either: if the download
# was interrupted, every segment's own transcription for the rest of that
# capture hits the same missing model, one at a time, rather than the
# capture failing fast up front.
# model = "base"

[dictation.privacy]
save_history = false
local_only = true
auto_clear_buffer = true
```

`local_only` accepts any provider that runs on this machine — all of
`parakeet-onnx`, `parakeet-mlx`, `lightning-whisper-mlx`, `faster-whisper`,
`qwen2audio`, `parakeet` and `canary`. Only a provider that sends audio off the
machine (`remote-whisper`) is substituted.

> **First run:** the Console loads the speech model *before* it opens the
> microphone, and the composer's voice chip says so ("Preparing speech
> model…"), with the details in a notification. On a fresh machine that first
> load downloads the model and can take several minutes; nothing is being
> recorded during it, the mic button cancels it, and a failure to load is
> reported as a model/provider problem rather than a microphone one.

### TTS Settings
```toml
[app_tts]
default_provider = "audio_cpp"
default_model_mode = "first_available"
default_voice_mode = "server_default"
default_format = "wav"
default_speed = 1.0

[app_tts.audio_cpp]
mode = "external"
base_url = "http://127.0.0.1:8080"
connect_timeout_seconds = 5
synthesis_timeout_seconds = 600
max_input_characters = 10000
max_response_bytes = 134217728
max_metadata_bytes = 1048576
max_catalog_models = 1000
max_voices_per_model = 1000
max_identifier_characters = 256
```

## Support & Feedback

- **Issues**: Report at GitHub repository
- **Feature Requests**: Submit via GitHub issues
- **Documentation**: Check Docs folder
- **Community**: Join discussions

---

**Last Updated**: 2026-07-29
**Version**: 2.4 (Console voice control V2: spoken commands and spoken feedback)
