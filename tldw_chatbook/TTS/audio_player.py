# audio_player.py
# Description: Simple audio playback for TTS - focusing on cross-platform compatibility
#
# Imports
import asyncio
import subprocess
import platform
import shutil
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
from loguru import logger

#######################################################################################################################
#
# Simple Audio Player for TUI

class PlaybackState(Enum):
    """Audio playback states"""
    IDLE = "idle"
    PLAYING = "playing"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class AudioPlayerInfo:
    """Information about current playback"""
    file_path: Optional[Path] = None
    state: PlaybackState = PlaybackState.IDLE
    process: Optional[subprocess.Popen] = None


class SimpleAudioPlayer:
    """
    Simple cross-platform audio player using system commands.
    
    For a TUI app, we don't need complex pause/resume - just play and stop.
    Uses native system commands for maximum compatibility.
    """
    
    def __init__(self):
        """Initialize audio player"""
        self._current: AudioPlayerInfo = AudioPlayerInfo()
        self._system = platform.system()
        self._lock = threading.Lock()  # Thread safety for state management
        self._find_player()
    
    def _find_player(self) -> None:
        """Find available audio player on the system"""
        if self._system == "Darwin":  # macOS
            self._player_cmd = ["/usr/bin/afplay"]
            self._player_name = "afplay"
        elif self._system == "Linux":
            # Try to find available player
            for player, cmd in [
                ("mpv", ["mpv", "--no-video", "--really-quiet"]),
                ("mplayer", ["mplayer", "-really-quiet"]),
                ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]),
                ("aplay", ["aplay", "-q"]),
                ("paplay", ["paplay"])
            ]:
                if shutil.which(player):
                    self._player_cmd = cmd
                    self._player_name = player
                    break
            else:
                self._player_cmd = None
                self._player_name = None
                logger.warning("No suitable audio player found on Linux")
        elif self._system == "Windows":
            # Windows has built-in support via wmplayer or start command
            # We'll determine the actual command in the play() method
            self._player_cmd = ["windows"]  # Placeholder, actual command built in play()
            self._player_name = "windows"
        else:
            self._player_cmd = None
            self._player_name = None
            logger.warning(f"Unsupported platform: {self._system}")
    
    def play(self, file_path: Path) -> bool:
        """
        Play an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if playback started successfully
        """
        # Stop any current playback
        self.stop()
        
        # Validate file
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"Audio file not found: {file_path}")
            return False
        
        # Check if we have a player
        if not self._player_cmd:
            logger.error("No audio player available")
            return False
        
        try:
            if self._system == "Windows":
                # Use Windows built-in wmplayer or start command instead of PowerShell
                # This avoids command injection risks
                wmplayer_path = Path("C:/Program Files/Windows Media Player/wmplayer.exe")
                if wmplayer_path.exists():
                    # Use Windows Media Player directly
                    cmd = [str(wmplayer_path), "/play", "/close", str(file_path)]
                else:
                    # Use start command to open with default application
                    # The empty string after 'start' is for the window title
                    cmd = ["cmd", "/c", "start", "", "/wait", str(file_path)]
            else:
                # Unix-like systems
                cmd = self._player_cmd + [str(file_path)]
            
            # Start playback in background
            with self._lock:
                self._current.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self._current.file_path = file_path
                self._current.state = PlaybackState.PLAYING
            
            # Start monitoring thread
            monitor = threading.Thread(target=self._monitor_playback, daemon=True)
            monitor.start()
            
            logger.info(f"Started playback of {file_path.name} using {self._player_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            self._current.state = PlaybackState.ERROR
            return False
    
    def stop(self) -> bool:
        """
        Stop current playback.
        
        Returns:
            True if stopped successfully
        """
        with self._lock:
            if self._current.process and self._current.state == PlaybackState.PLAYING:
                try:
                    self._current.process.terminate()
                    self._current.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._current.process.kill()
                except Exception as e:
                    logger.error(f"Error stopping playback: {e}")
                
                self._current.process = None
                self._current.state = PlaybackState.IDLE
                logger.info("Playback stopped")
                return True
        
        return False
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        with self._lock:
            return self._current.state == PlaybackState.PLAYING
    
    def get_state(self) -> PlaybackState:
        """Get current playback state"""
        with self._lock:
            return self._current.state
    
    def _monitor_playback(self) -> None:
        """Monitor playback process"""
        if self._current.process:
            self._current.process.wait()
            with self._lock:
                if self._current.state == PlaybackState.PLAYING:
                    self._current.state = PlaybackState.FINISHED
                    logger.debug("Playback finished")


# Global player instance
_audio_player_instance: Optional[SimpleAudioPlayer] = None

def get_audio_player() -> SimpleAudioPlayer:
    """Get the global audio player instance"""
    global _audio_player_instance
    if _audio_player_instance is None:
        _audio_player_instance = SimpleAudioPlayer()
    return _audio_player_instance


# Keep the original play_audio_file for backward compatibility
def play_audio_file(file_path: Path) -> bool:
    """
    Simple function to play an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if playback started
    """
    player = get_audio_player()
    return player.play(file_path)


# Async wrapper for Textual
class AsyncAudioPlayer:
    """Async wrapper for use in Textual"""
    
    def __init__(self):
        self._player = get_audio_player()
    
    async def play(self, file_path: Path) -> bool:
        """Play audio file asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._player.play, file_path)
    
    async def stop(self) -> bool:
        """Stop playback asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._player.stop)
    
    async def is_playing(self) -> bool:
        """Check if playing asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._player.is_playing)

#
# End of audio_player.py
#######################################################################################################################