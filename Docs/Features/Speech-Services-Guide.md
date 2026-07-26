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

[dictation.privacy]
save_history = false
local_only = true
auto_clear_buffer = true
```

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

**Last Updated**: 2026-07-26
**Version**: 2.2 (Native external audio.cpp Console speech)
