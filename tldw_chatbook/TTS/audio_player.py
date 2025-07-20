# audio_player.py
# Description: Simple audio playback for TTS - focusing on cross-platform compatibility
#
# Imports
import asyncio
import subprocess
import platform
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger

#######################################################################################################################
#
# Simple Audio Player for TUI

class PlaybackState(Enum):
    """Audio playback states"""
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class AudioPlayerInfo:
    """Information about current playback"""
    file_path: Optional[Path] = None
    state: PlaybackState = PlaybackState.IDLE
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None  # When playback started
    pause_time: Optional[float] = None  # When paused
    total_pause_duration: float = 0.0  # Total time spent paused
    duration: Optional[float] = None  # Total duration of the file
    position: float = 0.0  # Current playback position in seconds


class SimpleAudioPlayer:
    """
    Enhanced cross-platform audio player with pause/resume support.
    
    Uses native system commands with pause/resume capabilities where available.
    Falls back to stop/restart for players without native pause support.
    """
    
    def __init__(self):
        """Initialize audio player"""
        self._current: AudioPlayerInfo = AudioPlayerInfo()
        self._system = platform.system()
        self._lock = threading.Lock()  # Thread safety for state management
        self._supports_pause = False  # Whether the player supports native pause
        self._find_player()
    
    def _find_player(self) -> None:
        """Find available audio player on the system"""
        if self._system == "Darwin":  # macOS
            self._player_cmd = ["/usr/bin/afplay"]
            self._player_name = "afplay"
            self._supports_pause = False  # afplay doesn't support pause
        elif self._system == "Linux":
            # Try to find available player with pause support info
            for player, cmd, supports_pause in [
                ("mpv", ["mpv", "--no-video", "--really-quiet", "--input-ipc-server=/tmp/mpv-socket"], True),
                ("mplayer", ["mplayer", "-really-quiet", "-slave"], True),  # slave mode supports pause
                ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"], False),
                ("aplay", ["aplay", "-q"], False),
                ("paplay", ["paplay"], False)
            ]:
                if shutil.which(player):
                    self._player_cmd = cmd
                    self._player_name = player
                    self._supports_pause = supports_pause
                    break
            else:
                self._player_cmd = None
                self._player_name = None
                self._supports_pause = False
                logger.warning("No suitable audio player found on Linux")
        elif self._system == "Windows":
            # Windows Media Player supports pause through COM automation
            self._player_cmd = ["windows"]  # Placeholder, actual command built in play()
            self._player_name = "windows"
            self._supports_pause = True  # We'll use COM automation for pause
        else:
            self._player_cmd = None
            self._player_name = None
            self._supports_pause = False
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
                # Special handling for players with pause support
                if self._player_name == "mpv":
                    # Create a unique socket for this instance
                    import tempfile
                    socket_path = Path(tempfile.gettempdir()) / f"mpv-socket-{id(self)}"
                    cmd = self._player_cmd[:-1] + [f"--input-ipc-server={socket_path}", str(file_path)]
                    self._mpv_socket = socket_path
                    self._current.process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                elif self._player_name == "mplayer":
                    # For mplayer slave mode, we need pipes
                    self._current.process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    self._current.process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                
                self._current.file_path = file_path
                self._current.state = PlaybackState.PLAYING
                self._current.start_time = time.time()
                self._current.pause_time = None
                self._current.total_pause_duration = 0.0
                self._current.position = 0.0
            
            # Start monitoring thread
            monitor = threading.Thread(target=self._monitor_playback, daemon=True)
            monitor.start()
            
            logger.info(f"Started playback of {file_path.name} using {self._player_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            self._current.state = PlaybackState.ERROR
            return False
    
    def pause(self) -> bool:
        """
        Pause current playback.
        
        Returns:
            True if paused successfully
        """
        with self._lock:
            if self._current.state != PlaybackState.PLAYING:
                return False
            
            if not self._supports_pause:
                # For players without pause, we'll stop and track position
                self._current.position = self.get_position()
                self.stop()
                self._current.state = PlaybackState.PAUSED
                self._current.pause_time = time.time()
                logger.info(f"Paused at position {self._current.position:.1f}s (using stop)")
                return True
            
            # Native pause support
            try:
                if self._player_name == "mpv" and hasattr(self, '_mpv_socket'):
                    # Send pause command via socket
                    import socket
                    import json
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.connect(str(self._mpv_socket))
                        s.send(json.dumps({"command": ["set_property", "pause", True]}).encode() + b'\n')
                elif self._player_name == "mplayer" and self._current.process and self._current.process.stdin:
                    # Send pause command to mplayer
                    self._current.process.stdin.write(b'pause\n')
                    self._current.process.stdin.flush()
                else:
                    return False
                
                self._current.state = PlaybackState.PAUSED
                self._current.pause_time = time.time()
                logger.info("Playback paused")
                return True
                
            except Exception as e:
                logger.error(f"Failed to pause: {e}")
                return False
    
    def resume(self) -> bool:
        """
        Resume paused playback.
        
        Returns:
            True if resumed successfully
        """
        with self._lock:
            if self._current.state != PlaybackState.PAUSED or not self._current.file_path:
                return False
            
            if not self._supports_pause:
                # For players without pause, restart from saved position
                # This is a limitation - we can't truly resume from exact position
                logger.info(f"Resuming from start (pause not supported by {self._player_name})")
                self._current.total_pause_duration += time.time() - self._current.pause_time
                return self.play(self._current.file_path)
            
            # Native resume support
            try:
                if self._player_name == "mpv" and hasattr(self, '_mpv_socket'):
                    # Send resume command via socket
                    import socket
                    import json
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.connect(str(self._mpv_socket))
                        s.send(json.dumps({"command": ["set_property", "pause", False]}).encode() + b'\n')
                elif self._player_name == "mplayer" and self._current.process and self._current.process.stdin:
                    # Send pause command again to toggle
                    self._current.process.stdin.write(b'pause\n')
                    self._current.process.stdin.flush()
                else:
                    return False
                
                self._current.state = PlaybackState.PLAYING
                self._current.total_pause_duration += time.time() - self._current.pause_time
                self._current.pause_time = None
                logger.info("Playback resumed")
                return True
                
            except Exception as e:
                logger.error(f"Failed to resume: {e}")
                return False
    
    def stop(self) -> bool:
        """
        Stop current playback.
        
        Returns:
            True if stopped successfully
        """
        with self._lock:
            if self._current.process and self._current.state in [PlaybackState.PLAYING, PlaybackState.PAUSED]:
                try:
                    self._current.process.terminate()
                    self._current.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._current.process.kill()
                except Exception as e:
                    logger.error(f"Error stopping playback: {e}")
                
                # Clean up mpv socket if exists
                if self._player_name == "mpv" and hasattr(self, '_mpv_socket'):
                    try:
                        Path(self._mpv_socket).unlink(missing_ok=True)
                    except:
                        pass
                
                self._current.process = None
                self._current.state = PlaybackState.IDLE
                self._current.start_time = None
                self._current.pause_time = None
                self._current.position = 0.0
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
    
    def get_position(self) -> float:
        """
        Get current playback position in seconds.
        
        Returns:
            Current position in seconds
        """
        with self._lock:
            if self._current.state == PlaybackState.IDLE:
                return 0.0
            elif self._current.state == PlaybackState.PAUSED:
                return self._current.position
            elif self._current.state == PlaybackState.PLAYING and self._current.start_time:
                # Calculate current position based on elapsed time
                elapsed = time.time() - self._current.start_time - self._current.total_pause_duration
                return elapsed
            else:
                return self._current.position
    
    def get_duration(self) -> Optional[float]:
        """Get total duration of current file"""
        with self._lock:
            return self._current.duration
    
    def _monitor_playback(self) -> None:
        """Monitor playback process"""
        if self._current.process:
            self._current.process.wait()
            with self._lock:
                if self._current.state == PlaybackState.PLAYING:
                    self._current.state = PlaybackState.FINISHED
                    logger.debug("Playback finished")
    
    def cleanup(self) -> None:
        """Clean up resources - call this before app exit."""
        self.stop()
        # Clean up any remaining resources
        if hasattr(self, '_mpv_socket') and self._mpv_socket:
            try:
                Path(self._mpv_socket).unlink(missing_ok=True)
            except:
                pass
        logger.debug("Audio player cleaned up")


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
        self._executor = None
    
    def _get_executor(self):
        """Get or create thread pool executor"""
        if self._executor is None:
            import concurrent.futures
            # Create a daemon thread pool that won't block exit
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="AudioPlayer",
                initializer=lambda: None
            )
            # Mark threads as daemon
            import threading
            for thread in threading.enumerate():
                if thread.name.startswith("AudioPlayer"):
                    thread.daemon = True
        return self._executor
    
    async def play(self, file_path: Path) -> bool:
        """Play audio file asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.play, file_path)
    
    async def pause(self) -> bool:
        """Pause playback asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.pause)
    
    async def resume(self) -> bool:
        """Resume playback asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.resume)
    
    async def stop(self) -> bool:
        """Stop playback asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.stop)
    
    async def is_playing(self) -> bool:
        """Check if playing asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.is_playing)
    
    async def get_state(self) -> PlaybackState:
        """Get state asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.get_state)
    
    async def get_position(self) -> float:
        """Get playback position asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.get_position)
    
    async def get_duration(self) -> Optional[float]:
        """Get duration asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.get_duration)
    
    async def cleanup(self) -> None:
        """Clean up resources asynchronously"""
        await self.stop()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._get_executor(), self._player.cleanup)
        
        # Shutdown our executor
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

#
# End of audio_player.py
#######################################################################################################################