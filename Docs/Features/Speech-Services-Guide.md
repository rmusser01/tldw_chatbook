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
- **Provider Selection**: Choose from OpenAI, ElevenLabs, Kokoro (local), etc.
- **Voice Selection**: Pick from available voices for the selected provider
- **Format**: Choose output format (MP3, WAV, etc.)
- **Provider Settings**: Adjust provider-specific parameters

### TTS Settings

Configure default TTS options:

- Default provider and voice
- Audio format preferences
- API key management
- Voice blend creation (Kokoro only)

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
- **No Cloud Storage**: Audio never uploaded without permission
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
[tts]
default_provider = "openai"
default_voice = "alloy"
default_format = "mp3"
```

## Support & Feedback

- **Issues**: Report at GitHub repository
- **Feature Requests**: Submit via GitHub issues
- **Documentation**: Check Docs folder
- **Community**: Join discussions

---

**Last Updated**: 2025-07-30
**Version**: 2.0 (Improved Speech Services)