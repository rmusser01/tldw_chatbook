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
synchronization remain deferred. Voice profiles do not own provider connection
or process settings; configure those once in **Global Settings → Speech & TTS**.

Profiles are owned locally and contain generation choices, not provider
connection details. They exclude server origins, credentials and API keys,
binary or `server.json` paths, managed-process settings, message text, and
provider health observations. Names compare uniquely after Unicode
normalization and case folding. A saved edit must use the revision originally
loaded, so a concurrent change is reported as a conflict instead of being
silently overwritten.

#### Private clone-reference storage boundary

The profile database can now retain one canonical reference recording and its
transcript for a profile. The source path is never persisted. Reference audio
and transcript are local plaintext protected by owner-only filesystem access;
this is not encryption. Profile database backups contain the same sensitive
reference audio and transcript, so protect and share those backups accordingly.
Removal is best-effort deletion, not forensic erasure: copies may remain in
backups, filesystem snapshots, SQLite recovery data, or storage media. The
Windows privacy posture remains unverified until TASK-13208.

Profiles that contain a reference can use it for compatible Guided Managed
audio.cpp generation. Speech Lab provides the supported setup path for a
reviewed reference-required Guided model. Ordinary profile/card export remains
the safe default: reference-free profiles use wire version 1, while a profile
with a reference uses sanitized wire version 2 with an explicit
`reference: omitted` marker. Neither format contains reference metadata,
audio, transcript, digest, recipe provenance, or source paths. Do not treat an
ordinary profile export as a reference backup.

To intentionally transfer a saved clone voice on a supported POSIX host, open
the Voice Profile library and choose **Export → Export portable voice bundle**.
Chatbook warns that the file contains plaintext voice audio and transcript and
does not enable Continue until you acknowledge that warning. The bundle is a
strict four-file container (`manifest.json`, `profile.json`, `reference.wav`,
and `reference.txt`), is created owner-only, and never overwrites an existing
destination. Its checksums detect byte changes; they are not a signature,
speaker-identity check, authenticity proof, or consent proof.

**Import bundle** shows its own plaintext warning before opening the picker.
Chatbook validates the unchanged source as hostile input, then shows bounded
UUID, name, recipe, model, dependency, and conflict facts. You must explicitly
choose Create, exact Reuse, or Import as copy. Import never overwrites, assigns
a character, changes a default, installs a model, or retargets to another
recipe. If the exact dependency is absent, Create/Copy is available only after
you separately accept an inactive **Needs compatible model** result. Install or
configure that exact Guided package in audio.cpp Settings, apply the saved
Speech Lab configuration when prompted, then refresh the library. A migrated
profile that says **Recipe provenance unavailable** must be previewed/generated
and saved as a new profile before it can be bundled.

For clone generation, Chatbook freezes the exact profile/reference revision,
then verifies a reviewed clone-capable Guided recipe and the exact
application-owned managed process generation. Only after acceptance does it
create an opaque owner-private temporary WAV below Chatbook's user-data area.
The local request sends typed `voice_ref` and `reference_text` fields to that
owned child. The file remains available through audio-response cleanup and is
then removed before Chatbook releases the provider operation.

External audio.cpp servers and Managed setups using your own `server.json`
never receive a client-local reference path. A reference-bearing profile used
with either setup fails safely without switching provider, model, or voice. Use
a reviewed Guided Managed package that declares clone support, or edit/select a
profile that does not require a reference.

The temporary reference is local plaintext, not encrypted. On supported POSIX
hosts Chatbook uses owner-only directories, no-follow descriptor checks, and a
live ownership lock. Startup cleanup removes only recognized unlocked
Chatbook-owned operation directories; unknown paths and live owners are left
alone. Chatbook rechecks the path identity immediately before the request, but
audio.cpp must still open that pathname, so other processes running as your
same OS user are outside this isolation boundary. Child-output content is
suppressed for a process generation after its first clone admission so delayed
path or transcript echoes do not enter the diagnostics view. Windows managed
materialization remains deferred; use an External server there, which
intentionally cannot receive local references.

#### Create and assign a cloned Voice Profile

1. Configure a reviewed clone-capable package under **Global Settings → Speech
   & TTS → audio.cpp → Managed local server → Guided setup**. Saving remains
   passive and does not start audio.cpp.
2. Open **Speech Services → Playground**, select audio.cpp and the
   reference-required Guided model, then use **Start & Set Up Voice** if the
   child is not running.
3. Choose a bounded PCM16 WAV and enter the exact bounded transcript. Chatbook
   shows no source path and warns that the reference, transcript, profile
   database, and temporary request file are local plaintext, not encrypted.
4. Choose **Create Voice & Generate**. Chatbook starts or joins its one managed
   child lazily, verifies the exact recipe/model/process generation, creates a
   private short-lived request materialization, and keeps the previous playable
   result if this attempt fails. On success, use the normal **Play** control.
5. Choose **Save as Voice Profile**. Review the name, then either save it
   unassigned or continue to Roleplay. The saved profile uses the exact
   successful reference and transcript; Chatbook does not reopen your source
   file or copy the current selectors.
6. In Roleplay, select a character and explicitly choose the suggested Voice
   Profile. Opening the page or seeing the suggestion does not assign it and
   does not change the app-wide default.
7. Start or continue that character's Console session and use **Speak** on a
   response. The exact assigned profile revision wins over global defaults; the
   first request starts or joins the compatible child and produces the normal
   complete-WAV playback result.

Changing the message, assignment, profile/reference, Guided configuration, or
managed child before admission causes a stale/refusal result rather than a
silent provider/model/voice switch. A later edit affects the next Speak only.
Discarding/replacing the result or closing Chatbook removes Chatbook-owned
temporary audio and materializations; it never deletes the WAV you selected.

This workflow does not send client-local references to External audio.cpp or a
Managed setup that uses your own `server.json`. It does not export reference
bytes in ordinary profiles/cards, claim Windows bundle or managed-reference
parity, or enable a recipe that is not in the reviewed release-0.5.1 registry.

The profile store is now schema version 4. Migration advances every supported
older store through validated private candidates and retains stable downgrade
siblings at the applicable boundaries: `<profile-db>.pre-v3.sqlite3` and
`<profile-db>.pre-v4.sqlite3`. Version 3 references migrate without invented
recipe provenance. Older builds refuse the newer active database. To downgrade
to a version-3-capable build, fully close Chatbook, restore the stable
`<profile-db>.pre-v4.sqlite3` sibling as the configured profile database, then
start the older build. Accept that all post-migration profile changes and recipe
provenance will be lost. Never copy over an open profile database.

Bundle inspection, archive work, publication, and cleanup are owned by one
application service. On shutdown Chatbook first seals and joins that service,
then closes the profile repository, then closes TTS runtime ownership. A
restart that finds unexplained portability residue fails closed and asks you to
exit Chatbook and inspect the app-owned portability directory manually; it does
not guess that a pathname is safe to delete.

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

### Setting up audio.cpp

Open **Global Settings → Speech & TTS → audio.cpp** and choose the setup path
that matches who owns the server process and configuration:

- **External server** connects to a compatible `audiocpp_server` that you start
  and manage yourself. Chatbook never stops or adopts that process.
- **Managed local server → Guided setup — no JSON editing** launches and
  supervises one local process from a separately installed server and one or
  more reviewed model packages. Chatbook creates an owner-private runtime
  configuration only when a deliberate Speech Lab action needs it.
- **Managed local server → Use an existing server.json** launches and supervises one
  local process from a prebuilt binary and existing configuration that you
  provide. Chatbook does not edit that file.

Chatbook never downloads or updates `audiocpp_server`. Installing the server
and obtaining model files remain explicit user actions. Guided compatibility
claims apply only to the exact package recipes and server release shown in the
package review; finding another GGUF file does not imply support.

Managed local server is currently offered on qualified POSIX hosts. On
Windows, use External server until the native managed-process lifecycle is
qualified there.

#### External server

To connect to a server you manage yourself:

1. Start your compatible `audiocpp_server` and note its HTTP or HTTPS origin.
2. Select **External server** in the audio.cpp global settings.
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

#### Guided managed setup — no JSON editing

Guided setup is the shortest supported first-time path:

1. Install a compatible `audiocpp_server` separately and obtain a reviewed
   model package. The package review identifies the exact supported audio.cpp
   release, model family, and variant.
2. Select **Managed local server**, then **Guided setup — no JSON editing**.
   Choose **Use detected audiocpp_server** to fill the unsaved draft from the
   command search path, or use **Browse** to select the executable.
3. Choose **Add local package…** and select the package directory; selection
   starts the bounded scan. Review every accepted candidate. The review shows
   its exact family/variant, task capabilities,
   compatibility evidence, public model ID, a path-safe package summary, and
   lazy-loading/resident-memory behavior. Multiple matches are never selected
   silently; remove any package you do not want in the shared server.
   A standalone PocketTTS GGUF is registered but marked **voice setup
   required** because it does not include a usable voice embedding. Choose a
   text-ready Supertonic package as the default for the one-click first sample.
   Alternatively, choose **Open Model Library** to review and install a curated
   package into Chatbook's shared artifact store. Installation verifies the
   exact pinned package but remains inactive: it neither installs
   `audiocpp_server` nor starts audio.cpp. Return to Settings, review the added
   package in the still-unsaved draft, then choose Save. The catalog also names
   local-only variants; those must be supplied through **Add local package…**
   and are never presented as downloadable. Across the pinned 21-family,
   67-package inventory, 45 reviewed variants are downloadable, 8 approved
   variants are local-only, and 14 are explicitly unsupported; those states
   describe artifact availability separately from recipe support.
4. Choose the default Guided model and backend. **Auto** selects only the
   reviewed portable CPU baseline; it does not infer an accelerated backend
   from the host. An unavailable or unevidenced model/backend combination is
   rejected with recovery guidance.
5. Choose **Save Settings**. Saving revalidates the selected package identities
   and persists configuration, but it does not launch a process, open a socket,
   contact audio.cpp, generate a runtime `server.json`, synthesize speech, or
   modify model files. The panel reports **Configuration saved — ready to
   test**.
6. For a text-ready default, choose **Open Speech Lab & Hear a Sample**. A
   voice-required default instead offers **Open Speech Lab to Test
   Connection** and does not promise generation. The handoff focuses Speech
   Lab's one current primary action, which may be **Start & Generate Sample**,
   **Restart & Apply Settings**, **Retry Sample**, or **Test Connection**.
   Inflect Micro v2 additionally requires eSpeak-ng with English data. Before
   testing, verify that the server process can resolve both by their default
   names; installation alone does not prove loader/data discoverability. Guided
   setup does not discover or persist private library/data paths. If explicit
   paths are required, use an advanced `server.json` you control and follow the
   pinned upstream Inflect session-option guidance.
7. Run **Start & Generate Sample**. Chatbook lazily creates one private,
   generation-local configuration, starts one child, verifies the selected
   catalog entry, and generates one complete validated WAV.
8. The current result exposes **Play/Pause**, duration and validation status,
   safe provider/model/voice/configuration/process provenance, **Generate
   again**, and **Save WAV**. Optional automatic playback is a separate Speech
   Studio preference; Guided setup never changes it.

All accepted models are registered in the same lazy child. Selecting another
accepted model reuses that child rather than launching a second server. A model
loaded by audio.cpp may remain resident in memory until **Shut down server**;
the interface does not promise per-model unloading.

Model Library removal first previews every Settings, profile, character,
runtime, and shared-store owner. Busy or referenced packages remain installed
until every blocker has an explicit resolution. If interruption or a changed
package prevents commit, reopen the preview after the active operation ends;
Chatbook rechecks the exact fingerprint and never silently retargets a model or
deletes a private clone reference as a side effect.

#### Managed local server with your server.json

The manual source is for a local build and configuration you trust. Review the
executable and JSON provenance before allowing Chatbook to run them, then:

1. Make sure you already have a compatible prebuilt `audiocpp_server` binary
   and its existing `server.json`. Chatbook never downloads, updates, or edits
   either artifact.
2. Select **Managed local server**, then **Use an existing server.json**.
   Choose **Use detected audiocpp_server** to fill the draft from your command
   search path, or use **Browse** to select the executable yourself. Detection
   changes only the unsaved form.
3. Browse to the existing `server.json`. It must be a strict JSON object that
   binds exactly to `127.0.0.1` and declares an unused port. Remote or wildcard
   hosts are intentionally rejected in Managed mode.
4. Check relative paths in `server.json` yourself. Chatbook launches the child
   with the JSON file's directory as its working directory, so relative model
   and resource paths resolve from there.
5. Review **Advanced** for startup, health-check, shutdown, timeout, response,
   and catalog safety bounds, then choose **Save Settings**. Saving validates
   the active Managed fields and persists them; it does not launch, probe, or
   contact audio.cpp.
6. Open **Speech Services → Playground**, select audio.cpp, and inspect the
   runtime card. Choose **Start & Test Connection**. The asynchronous action
   lazily launches one child, waits for its speech contract, and refreshes the
   configured multi-model catalog.
7. Select a model and optional voice, enter text, and choose **Generate
   Speech**. Chatbook validates the complete WAV before exposing **Play** and
   **Export**.

The runtime card separates durable and live truth. **Saved generation** is the
latest configuration on disk, **Active generation** is the configuration held
by the current adapter, and **Process generation** identifies the exact child.
If you save new Managed settings while speech or a child is active, that child
keeps its immutable launch configuration and the card shows the new settings
as pending. Choose **Restart & Apply Settings** to drain admitted speech, stop
that exact child, apply the latest saved generation, and start one replacement.

**Shut down server** is a process action and is distinct from playback
**Stop**. Shutdown drains and reaps the managed child without immediately
starting another one. A later deliberate Start/Test, catalog refresh, or
generation request starts it lazily again. Merely opening Settings, mounting
the Playground, expanding details, or reading diagnostics never launches it.

If a managed child exits unexpectedly, Chatbook marks its endpoint unavailable.
If the live child becomes unhealthy, Chatbook keeps its endpoint visible but
marks its catalog evidence stale. Neither case triggers a restart loop. Correct
the binary/configuration or port problem if needed, then use **Start & Test
Connection** or **Restart**. The collapsed **Recent managed diagnostics**
section contains only bounded, best-effort-sanitized in-memory child output.
Treat it as potentially sensitive; it is cleared when a new process generation
starts and is never saved to provider settings.

To return to a server you manage yourself, select **External server**, enter its
origin, and save. If a managed child is still active, Speech Lab shows the
pending handoff; choose **Apply Settings & Stop Managed Server**. The next
External test or generation contacts only the saved external origin.

Global Settings owns connection details, reviewed Guided packages, and global
defaults used by ordinary consumers such as Console. **Speech Studio
preferences** are a separate, Studio-only layer: they may override selections
and optional auto-play for Studio without changing Global Settings. Speech Lab
links back to Global Settings instead of duplicating durable connection fields.

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

External mode preserves its ownership boundary: application shutdown leaves a
user-owned server running. Managed mode stops only the exact child Chatbook
launched and never scans for, adopts, or signals another process.

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
| "send" | capture-ending | Ends the capture, inserts the text, and sends the message once insertion has completed. Refused if you switched tabs while it was transcribing (the text is in the *original* tab's draft, so sending would ship the other tab's), or if Send is blocked for any other reason — the refusal says which. **In the hands-free loop:** while listening or counting down, this drives an immediate send (same as the countdown expiring on its own). Said while a reply is already outstanding (only reachable in acoustic mode, the only mode that keeps the mic open mid-reply) it cannot start a second turn on top of the first — the capture still ends, but hands-free exits instead of silently doing nothing. |
| "discard" | capture-ending | Ends the capture without inserting anything — the same as pressing Cancel. No confirmation is asked; saying it is treated as explicit intent. Once the capture has moved on to transcribing there is nothing left to abort, so it is refused with "Too late to discard" and the text still lands. **In the hands-free loop, this also exits it** — see "Entering and exiting" above. |
| "read that back" | capture-ending | Ends the capture (inserting the text first), then speaks the latest **completed** assistant reply. If the reply is still streaming, or there is none yet, it acknowledges instead of speaking a partial answer. **In the hands-free loop, this also exits it** — see "Entering and exiting" above. |
| "new session" | capture-ending | Ends the capture (inserting the text first), then opens a new session tab. **In the hands-free loop, this also exits it** — see "Entering and exiting" above. |
| "hands free" | inline | Enters the hands-free conversation loop, adopting the still-open capture as its first turn — capture keeps running. See "Hands-Free Conversation Loop" below. |

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

**Onset pre-roll.** VAD gating a frame at a time can clip the very start of
an utterance: low-energy onsets — word-initial fricatives especially ("s" in
"stop"/"send") — are classified as non-speech at the default
`vad_aggressiveness`, so without further help the first frame(s) of a word
would be dropped before transcription ever saw them (observed live as "stop"
transcribed as "top"/"dot"-like forms, "send" as "and"). The recorder guards
against this with a small pre-roll: it keeps the last
`dictation.vad_preroll_ms` (default 240 ms / 12 frames) of *rejected* audio
on hand and replays it, through the same path as any accepted frame, the
instant VAD accepts a frame after a silence run — recovering the clipped
onset without holding back enough rejected audio to meaningfully dilute VAD
gating. It never fires between two already-accepted frames, so ongoing
speech and the choreography above are unaffected.

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

**Streaming playback.** Spoken feedback (and Console **Speak**, same speech
pipeline) can play some responses live through the audio device instead of
waiting on a finished file.

*What streams today.* Only the audio.cpp adapter's response is eligible —
its complete PCM16-WAV body validates as a playable stream. tldw_chatbook
still writes that response to a temporary artifact exactly as before, but
the instant it validates, playback moves to the live device and the
now-redundant temp file is discarded immediately instead of being kept for
file-based playback. That makes playback interruptible at the device — a
new capture or a new utterance cuts audio within roughly two audio blocks
instead of waiting on a file — and leaves nothing on disk to replay or
export for that turn. Latency to first sound is unchanged: audio.cpp still
delivers one complete WAV per request, not incremental chunks, so there is
no "starts talking sooner" win here, only the interruptibility one.

*What still falls back byte-identically to the pre-streaming path* — same
temp file, same file-based playback, same everything:

- Every legacy-bridge provider (openai, elevenlabs, kokoro, chatterbox,
  alltalk, higgs, ...) — that bridge never populates a response sample
  rate, and without one the sink cannot open. Unblocking these is a filed,
  three-leg follow-up (TASK-1880) (plumb a sample rate onto legacy-bridge responses,
  add a caller-scoped raw-PCM request option, and add a PCM-safe legacy
  fallback) — none of it has shipped yet, so do not expect these providers
  to stream.
- Any compressed format (MP3, Opus, AAC, FLAC).
- `sounddevice` not installed, or the sink otherwise unavailable.
- A device failure at open or mid-stream.

*Numbers.* Playback becomes audible once 300 ms of audio is buffered; an
interrupting stop reaches silence within 2 audio blocks — about 40 ms at
the default 20 ms block size — by aborting the output stream rather than
draining it.

*No configuration changed.* `dictation.spoken_feedback` and the `[app_tts]`
settings above behave exactly as documented; this is an internal delivery
upgrade for responses the sink already recognized as safe, not a new
setting.

*For test authors.* Every test is guarded against opening a real audio
device by default (`Tests/conftest.py`'s autouse `_no_real_audio_device`
fixture patches out the `sounddevice` import); a test that genuinely needs
real hardware must opt out with `@pytest.mark.real_audio_device`.

### Hands-Free Conversation Loop

Composed from the pieces above (Console dictation, the spoken-command
grammar, and streaming reply speech): speak, pause, it sends, the reply is
spoken back sentence by sentence, and the microphone reopens automatically
for your next turn — a full voice-in/voice-out conversation with no keyboard
or mouse required once it's running.

**Entering and exiting.** Say **"Console, hands free."** while a capture is
already open (the current capture becomes the loop's first turn — nothing
you already dictated is lost), or press **`Ctrl+Shift+H`** at any time, from an
open capture or from idle (idle opens a fresh capture).

To leave the loop, **`Ctrl+Shift+H`** again, **Esc**, or the mic button work from
**any point** in the loop (mid-listen, mid-countdown, while the reply is
still generating, or while it is being spoken) and return the Console to
its ordinary, pre-loop behavior. While the loop is running, **Esc takes
priority over any widget-level Esc binding elsewhere on the screen** (e.g.
the transcript's own clear-selection) so it reliably exits from wherever
your focus happens to be; outside the loop this priority is inert and Esc
behaves exactly as it always has.

Four spoken commands also end the loop — **"Console, stop."**, **"Console,
discard."**, **"Console, new session."**, and **"Console, read that
back."** — but, like every spoken command, they need an open microphone to
be heard. In the default (non-acoustic) mode the mic is open only while
listening or counting down, not while the reply is generating or being
spoken, so these four are reachable during those two states only;
acoustic mode (see "Barge-in" below) reopens the mic as soon as the reply
starts generating, so it widens their availability to the whole turn. The
last three exit as a side effect of what they otherwise do: none of them
continues the same conversation (discarding throws away what you just
said, a new session switches tabs, and reading back speaks an
*already-completed* reply rather than starting a new turn), so hands-free
ends rather than being left running with nothing to listen for. There is
an **eighth** way the loop ends: saying **"Console, send."** while a reply
is already outstanding cannot start a second turn on top of the first —
there is no way to interleave them — so the capture still ends and
hands-free exits rather than silently doing nothing.

**How a turn works.** Speak normally; when you pause, the composer's voice
chip counts down ("hands-free · sending in 1.5s…") before sending — say
anything else, or press any key, and the countdown cancels and you keep
listening instead. Once it sends, the chip shows "hands-free · thinking…"
while the reply generates, then "hands-free · speaking" once it starts
talking back, sentence by sentence, through your speakers. When the reply
finishes, the chip returns to the ordinary recording indicator and the
microphone is live again for your next turn.

**Honest timing.** The pause-to-send delay is not instantaneous — it is the
sum of three real steps: the dictation silence gate
(`dictation.silence_threshold_seconds`, default 2.0 s) plus that segment's
own transcription (roughly 0.3–1 s for a short utterance on a warm local
model) plus the hands-free countdown itself
(`dictation.handsfree_send_delay_seconds`, default 1.5 s, cancellable the
entire time). Budget **around 4 seconds** from the moment you stop talking
to the moment your message actually sends — this is the same
silence-gate-plus-transcription cost the ordinary spoken-command grammar
above pays, with the countdown added on top so you have a visible, audible
window to change your mind.

**Barge-in (interrupting a reply).** By default, hands-free relies on a
**keyboard barge-in**: press any key while the reply is speaking (or still
generating) and it silences immediately — you keep whatever was already
generated in the transcript, only the audio stops, and the microphone
reopens right away for you to speak your next turn. There is no acoustic
echo cancellation in this app, so **spoken barge-in is opt-in**
(`dictation.acoustic_barge_in`, default `false`) and comes with a real
trade-off: with it on, the microphone reopens the instant the reply starts
generating rather than waiting for it to finish, and speaking over the
reply silences it exactly like a keypress would — but on speakers, without
echo cancellation, the recognizer will pick up the reply's OWN voice coming
out of your speakers and try to transcribe it. **Headphones are strongly
recommended whenever acoustic barge-in is enabled** — on speakers, expect
false "speech" detection from the reply itself.

**If the room goes quiet.** A capture that hits its own service-side limits
(the 60 s wall-clock cutoff, or the recorder's buffer cap) with nothing
dictated reopens once for a fresh turn rather than ending the loop outright
— but a *second consecutive* empty-limit ending exits the loop rather than
reopening forever. In practice this means an unattended hands-free session
in a silent room exits on its own after roughly **two minutes** (two
back-to-back 60 s captures with nothing said), rather than leaving the
microphone open indefinitely.

**Degraded mode (no `webrtcvad`).** Everything above — the countdown, the
silence-based auto-send, and spoken/acoustic barge-in — depends on the
recorder's voice-activity detection. Without the optional `webrtcvad-wheels`
package installed, hands-free still opens the microphone and dictates, but
the pause-to-send countdown and any voice-triggered barge-in never fire (the
same limitation the plain Console dictation capture already has in this
mode — see "Choreography and latency" above); a warning explains this the
moment you enter the loop in that state. Use the mic button, Esc, `Ctrl+Shift+H`,
or spoken "Console, stop." to end a turn manually instead.

**What actually speaks the reply.** Reply speech goes out through each
provider's existing synthesis path, sentence by sentence, exactly like any
other Console TTS request — there is no separate "hands-free voice"
pipeline. The audio.cpp adapter streams its response live to the audio
device the same way ordinary spoken feedback does (see "Streaming playback"
above); every other provider plays back a completed audio file per
sentence, the same as it always has. Reply speech is intrinsic to the
loop and does not read `dictation.spoken_feedback` — that setting only
governs status acknowledgements ("Sent.", "Discarded.", ...) outside the
loop; hands-free speaks its replies regardless of how it is set.

**Configuration:**
```toml
[dictation]
# Seconds a finalized segment sits in the hands-free countdown before it
# auto-sends. Cancellable the whole time by speaking again or pressing any
# key. Must be a finite, positive number; invalid or non-positive values
# fall back to this 1.5s default.
handsfree_send_delay_seconds = 1.5
# Opt-in acoustic barge-in: lets spoken interruption (not just a keypress)
# silence a reply mid-speech, and reopens the microphone as soon as the
# reply starts generating rather than waiting for it to finish. Off by
# default -- there is no echo cancellation, so enabling this on speakers
# (without headphones) risks the recognizer transcribing the reply's own
# voice. See "Barge-in (interrupting a reply)" above.
acoustic_barge_in = false
```

**Out of scope (for now).** Wake-word activation (hands-free is entered by
spoken command or keypress only), acoustic echo cancellation, a Settings UI
for the two keys above (edit `config.toml` directly), and speaking replies
outside the loop (the existing per-message Speak affordance and
`spoken_feedback` toggle are unrelated and unaffected).

### Realtime engine

An optional low-latency engine for the same hands-free loop above: instead
of the pipeline (record → silence-gate → transcribe → LLM → sentence-by-
sentence TTS), it holds one live connection to a provider's realtime speech
API. Audio streams both directions continuously, the server detects your
turns, and replies begin in well under a second instead of the pipeline's
~4 s pause-to-send budget — same loop, a different engine underneath.
Entry and exit are unchanged and **engine-transparent**: **`Ctrl+Shift+H`**
starts and ends the loop exactly as described above regardless of which
engine is running. The chip names the engine while it runs — `realtime ·
connecting…`, `realtime · listening`, `realtime · thinking…`, `realtime ·
speaking`, and `realtime · reconnecting…` — so it is always visible which
engine you are actually talking to.

**Off by default, and only OpenAI Realtime today.** Enabling it is an
explicit opt-in (`realtime.enabled`, never inferred from an API key merely
being present); OpenAI's Realtime API is the only supported provider in
this release.

**Honest UX difference: spoken commands do not exist inside realtime
mode.** Realtime mode runs no client-side speech-to-text at all — the
provider's model does the hearing directly from the raw audio stream — so
none of the "Console, …" spoken commands described above, including
"Console, stop.", can be heard or classified while it is active. **Exits
are key/mouse only**: `Ctrl+Shift+H`, Esc, or the mic button.

**Privacy: microphone audio streams continuously, not just after a
pause.** This is the sharpest difference from the pipeline engine above,
which only sends audio once a pause is detected. In realtime mode, once
the session is live, microphone audio streams to the provider
continuously for the *entire* session — every word, immediately, for as
long as the loop is running — subject only to barge-in gating (in the
default speaker-safe mode, outbound audio is muted client-side while a
reply is playing; see "Barge-in" below). This is not "listens after you
pause": it is a continuously open microphone stream to a third-party
provider for the whole live session.

**Cost: billed per connected minute, not per message.** Realtime sessions
are billed by the provider for however long the session stays connected,
not per message or per token the way the pipeline engine's STT/LLM/TTS
calls are. An idle session left connected keeps costing money even if
nobody is talking. The **idle timeout** below (`realtime.idle_timeout_
minutes`, default 5) protects against this: no activity — no turn sent
and no reply finishing — for that many minutes ends the loop
automatically with a toast naming the reason. It never fires while a
reply is actively being spoken, so a long answer is never cut off by the
cost guard.

**Turn detection: how the provider decides you have finished speaking.**
Two modes, `realtime.turn_detection`:

- **`semantic_vad`** (default) — the model decides from *what you said*
  whether your turn is over. This is the default because the alternative's
  provider defaults are aggressive: server VAD ends a turn after **200 ms**
  of quiet at a fixed energy threshold, so in an ordinary room a
  mid-sentence pause, a keystroke (barge-in here *is* a keypress, right
  next to a live microphone), or a cough can each end your turn. Each
  fragment is then transcribed on its own, and a speech model handed half a
  syllable of keyboard noise reports words you never said. If your
  transcripts show random or invented words rather than your question, this
  is the setting to check first.
- **`server_vad`** — silence-gated, and tunable. Choose it when you want
  direct control of the gate: `realtime.vad_threshold` (0–1, how loud
  counts as speech) and `realtime.vad_silence_ms` (how long a pause ends
  the turn; raise it if your speech is being cut off mid-thought). Both are
  optional — leave them unset to use the provider's own values. Both apply
  to `server_vad` **only**: the API rejects them outright under
  `semantic_vad`, so the app drops them in that mode rather than sending a
  request that would fail.

All three are editable in **Settings → Speech & TTS → Realtime engine**,
where the two numbers are disabled unless server VAD is selected.

**Barge-in.** The same two modes as the pipeline engine, controlled by the
same `dictation.acoustic_barge_in` key: by default (speaker-safe) outbound
microphone audio is gated client-side while a reply plays, and any
keypress cuts the reply and reopens the mic; with acoustic barge-in
enabled, the microphone stays open the whole time and speaking over a
reply cuts it, with the same no-echo-cancellation / headphones warning as
the pipeline engine.

**Reconnect and failure.** An unexpected connection drop gets **one**
automatic reconnect attempt (a fresh connection, reseeded from the visible
conversation, with a toast confirming "reconnected"). A second failure in
the same loop entry exits loudly with the reason — there is no silent
retry loop and no silent death. **Connect failures fall back loudly, and
only to a genuinely usable pipeline:** if the realtime engine can't
connect at all and the pipeline engine is usable (STT installed, VAD not
degraded), hands-free enters the pipeline loop instead, with a toast
naming the realtime failure so the switch is never silent. If the
pipeline isn't usable either, both reasons are toasted and the loop does
not open — you are never dropped into a dead capture.

**Continuity.** Voice turns land in the Console transcript as ordinary
messages, same as the pipeline engine. Entering the loop seeds the
realtime session from the conversation's existing history (and reseeds it
on reconnect) so context carries over both ways.

**Configuration:**

Install the extra first — it carries the WebSocket transport plus the
audio capture/playback backends the realtime engine needs, none of which
come with a plain install:

```bash
pip install -e ".[realtime]"        # from a checkout
pip install "tldw_chatbook[realtime]"   # from a release
```

```toml
[realtime]
enabled = true                # off by default; explicit opt-in only
provider = "openai"           # the only supported value today
# model = "gpt-realtime"      # optional override; OpenAI's current default
# voice = "marin"             # optional; unset uses the provider's default voice
idle_timeout_minutes = 5      # cost guard -- ends an unattended session
turn_detection = "semantic_vad"  # or "server_vad"; see "Turn detection" above
# vad_threshold = 0.6         # server_vad only; unset = provider default
# vad_silence_ms = 700        # server_vad only; unset = provider default (200 ms)

[dictation]
# "auto" uses realtime iff realtime.enabled, "pipeline" forces the engine
# above without deleting the [realtime] block, "realtime" forces realtime
# (a loud, refused toast if it isn't actually enabled -- never a silent
# downgrade back to the pipeline).
handsfree_engine = "auto"
```

All of the above is also editable from **Settings → Speech & TTS →
Realtime engine**: the enable switch, provider (OpenAI only), model,
voice, idle timeout, and hands-free engine override all live there, and
Save persists them the same way the rest of that screen's fields do.

**Out of scope (for now).** Other realtime providers (Gemini Live is a
named follow-up behind the same protocol seam), tool/agent calls inside a
realtime session, WebRTC transport, a per-session voice picker, and
feeding the Console cost chip from realtime usage.

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

### Console Hands-Free Loop
- `Ctrl+Shift+H`: Enter the hands-free conversation loop (from idle or an open
  capture), or exit it if already running.
- `Esc` / mic button / spoken "Console, stop.": Exit the loop from any
  state — see "Hands-Free Conversation Loop" above.

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
# How many milliseconds of recently-rejected audio the recorder replays the
# instant VAD accepts a frame after a silence run, to recover a clipped
# speech onset (see "Onset pre-roll" above). Must be a non-negative integer;
# invalid values fall back to this 240ms (12-frame) default.
vad_preroll_ms = 240
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
# Hands-free conversation loop -- see "Hands-Free Conversation Loop" above
# for the full behavior. Countdown duration (seconds) before an auto-send;
# invalid or non-positive values fall back to this 1.5s default.
handsfree_send_delay_seconds = 1.5
# Opt-in spoken barge-in for the hands-free loop; headphones recommended
# when enabled (no echo cancellation). Default false.
acoustic_barge_in = false

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

Use **Global Settings → Speech & TTS → audio.cpp** to switch to Managed mode
and choose the executable and `server.json`; the UI validates those artifacts
without launching them. Dormant External and Managed values are retained when
you switch modes, but only the selected mode is applied at runtime.

## Support & Feedback

- **Issues**: Report at GitHub repository
- **Feature Requests**: Submit via GitHub issues
- **Documentation**: Check Docs folder
- **Community**: Join discussions

---

**Last Updated**: 2026-08-02
**Version**: 2.5 (streaming spoken-feedback playback for audio.cpp responses)
