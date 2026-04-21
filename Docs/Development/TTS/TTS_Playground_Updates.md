# TTS Playground Updates Summary

## Overview
Updated the TTS Playground dropdowns to accurately reflect the options available for each TTS provider/backend based on official documentation.

## Changes Made

### 1. OpenAI TTS
- **Updated Voices**: Added new voices (ash, ballad, coral, sage, verse) to the existing set
- **Models**: Kept tts-1 and tts-1-hd (correct as is)
- **Formats**: All formats already available (mp3, opus, aac, flac, wav, pcm)

### 2. ElevenLabs TTS
- **Updated Models**: Added Flash models for low-latency applications:
  - eleven_flash_v2 (Low Latency)
  - eleven_flash_v2_5 (Ultra Low Latency)
  - Kept existing models (monolingual v1, multilingual v1/v2, turbo v2)
- **Added Provider-Specific Settings**:
  - Voice Stability (0.0-1.0)
  - Similarity Boost (0.0-1.0)
  - Style (0.0-1.0)
  - Speaker Boost (boolean switch)
- These settings are shown/hidden dynamically based on provider selection

### 3. Kokoro TTS
- **Expanded Voice List**: Added all available voices:
  - American Female (11 voices): alloy, aoede, bella, heart, jessica, kore, nicole, nova, river, sarah, sky
  - American Male (2 voices): adam, michael
  - British Female (2 voices): emma, isabella
  - British Male (2 voices): george, lewis
- **Added Language Selection**: Dropdown for supported languages:
  - American English (a)
  - British English (b)
  - Japanese (j)
  - Mandarin Chinese (z)
  - Spanish (e)
  - French (f)
  - Hindi (h)
  - Italian (i)
  - Brazilian Portuguese (p)
- Language selection is shown/hidden dynamically for Kokoro only

## UI Improvements

### Dynamic Provider Settings
- Provider-specific settings are now shown/hidden based on the selected provider
- Kokoro shows language selection
- ElevenLabs shows voice tuning parameters
- OpenAI shows only standard settings

### Event System Updates
- Updated `STTSPlaygroundGenerateEvent` to accept `extra_params` dictionary
- Event handler now passes provider-specific settings to the TTS backends
- Added logging for extra parameters in the generation log

## Technical Implementation

### CSS Updates
- Added visibility controls for provider-specific sections
- `#kokoro-language-row` - Hidden by default, shown for Kokoro
- `#elevenlabs-settings` - Hidden by default, shown for ElevenLabs

### Form Data Collection
- `_generate_tts()` method now collects provider-specific settings
- Extra parameters are passed through the event system
- Settings are logged for debugging

## Testing Notes
- All UI elements render correctly
- Provider switching updates the appropriate dropdowns
- Settings are properly collected and passed to the event system
- No errors on tab selection or provider switching

## Future Considerations
- The backend services may need updates to fully utilize the ElevenLabs voice settings
- Additional Kokoro voices for other languages can be added as they become available
- Consider adding tooltips to explain each setting's purpose