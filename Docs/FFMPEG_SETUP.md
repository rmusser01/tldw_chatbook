# FFmpeg Setup for tldw_chatbook

## Issue
The application is encountering FFmpeg library loading errors when using TTS features with certain backends. The error messages indicate missing FFmpeg dynamic libraries.

## Error Details
```
OSError: dlopen(...libtorio_ffmpeg6.so, 0x0006): Library not loaded: @rpath/libavutil.58.dylib
```

This occurs because the `torio` library (part of PyTorch audio processing) expects specific versions of FFmpeg libraries.

## Solution

### macOS (Homebrew)
```bash
# Install FFmpeg
brew install ffmpeg

# If you encounter version mismatch issues, you may need to install specific versions
# Check which version torio expects and install accordingly
brew list --versions ffmpeg

# Create symlinks if needed (example for version mismatch)
# Note: Only do this if you understand the risks of version mismatches
# cd /opt/homebrew/lib
# ln -s libavutil.57.dylib libavutil.58.dylib  # Example only
```

### Linux (Ubuntu/Debian)
```bash
# Install FFmpeg
sudo apt update
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavutil-dev

# For specific versions
sudo apt install ffmpeg=7:4.4.* # Example for version 4.4
```

### Alternative Solution - Disable Problematic TTS Backends
If you don't need the TTS features that require FFmpeg, you can disable them in the configuration:

1. Edit your config file at `~/.config/tldw_cli/config.toml`
2. Disable backends that require FFmpeg (like local_kokoro_*, local_higgs_*)
3. Use backends that don't require FFmpeg (like openai_official_*, elevenlabs_*)

## Verification
After installing FFmpeg, verify it's working:
```bash
ffmpeg -version
```

Then restart the tldw_chatbook application.

## Note
The FFmpeg loading errors are warnings and won't prevent the core functionality of the application from working. They only affect certain TTS backends that rely on audio processing capabilities.