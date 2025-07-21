# TTS Playground Audio Fix Summary

## Issues Fixed

1. **Audio stops playing after changing voices**
   - Root cause: The audio player wasn't properly cleaning up previous playback before starting new audio
   - Solution: Added explicit `await self.app.audio_player.stop()` before playing new audio in `_play_audio_async`

2. **Pause/Stop buttons not working**
   - Root cause: All audio workers were using `exclusive=True`, preventing pause/stop from running while play was active
   - Solution: Changed pause and stop workers to use `exclusive=False` so they can interrupt playback

## Changes Made

### tldw_chatbook/UI/STTS_Window.py
1. Modified `_play_audio_async()` to explicitly stop any existing playback before starting new audio
2. Changed `_pause_audio()` to use `exclusive=False` for the worker
3. Changed `_stop_audio()` to use `exclusive=False` for the worker

### tldw_chatbook/TTS/audio_player.py
1. Enhanced `stop()` method to handle macOS afplay more aggressively:
   - Uses `kill()` directly for afplay on macOS instead of `terminate()`
   - Added better timeout handling and force kill if needed
2. Improved `_monitor_playback()` to handle interruptions gracefully

## Technical Details

The main issue was that Textual's exclusive workers prevented audio control operations from running while audio was playing. On macOS, `afplay` blocks until audio finishes, so the play worker would hold the exclusive lock for the entire duration of playback.

By making pause/stop workers non-exclusive, they can now interrupt the playing audio properly. The enhanced process termination ensures afplay processes are killed immediately on macOS.

## Testing

To test the fixes:
1. Generate a TTS sample with one voice
2. While it's playing, try pause/stop buttons - they should work immediately
3. Change to a different voice and generate again - the new audio should play
4. Previous audio should stop automatically when playing new audio