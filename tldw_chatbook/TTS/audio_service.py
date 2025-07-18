# audio_service.py
# Description: Audio processing and format conversion service for TTS
#
# Imports
import asyncio
import io
import os
from typing import Optional, Dict, Any, Union, BinaryIO
from pathlib import Path
import tempfile
from loguru import logger

# Optional numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    logger.warning("numpy not available. Some audio processing features will be limited.")

# Third-party imports
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available. Some audio format conversions will be limited.")

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available. Some audio format conversions will be limited.")

#######################################################################################################################
#
# Audio Service Implementation

class AudioService:
    """Service for audio format conversion and processing"""
    
    # Audio format specifications
    SAMPLE_RATES = {
        "mp3": 44100,
        "opus": 48000,
        "aac": 44100,
        "flac": 44100,
        "wav": 44100,
        "pcm": 24000,  # Default for TTS
        "m4a": 44100,
        "m4b": 44100,  # M4B is essentially M4A with chapter markers
    }
    
    BIT_RATES = {
        "mp3": "192k",
        "opus": "128k",
        "aac": "192k",
        "m4a": "192k",
        "m4b": "128k",  # Typically lower for audiobooks
    }
    
    def __init__(self):
        """Initialize audio service"""
        self.temp_dir = tempfile.gettempdir()
    
    async def convert_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        target_format: str,
        source_format: Optional[str] = None,
        sample_rate: Optional[int] = None
    ) -> bytes:
        """
        Convert audio data to target format.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            target_format: Target audio format (mp3, opus, aac, flac, wav, pcm)
            source_format: Source format (optional, will be detected if not provided)
            sample_rate: Sample rate (optional, uses format default if not provided)
            
        Returns:
            Converted audio data as bytes
        """
        # Validate target format
        target_format = target_format.lower()
        if target_format not in self.SAMPLE_RATES:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        # Handle numpy array input (raw PCM)
        if NUMPY_AVAILABLE and isinstance(audio_data, np.ndarray):
            if source_format is None:
                source_format = "pcm"
            if sample_rate is None:
                sample_rate = self.SAMPLE_RATES.get(source_format, 24000)
            audio_data = self._numpy_to_bytes(audio_data)
        
        # If already in target format and not numpy, return as-is
        if source_format == target_format and isinstance(audio_data, bytes):
            return audio_data
        
        # Use appropriate conversion method
        if PYDUB_AVAILABLE:
            return await self._convert_with_pydub(
                audio_data, target_format, source_format, sample_rate
            )
        elif SOUNDFILE_AVAILABLE and target_format in ["wav", "flac"]:
            return await self._convert_with_soundfile(
                audio_data, target_format, source_format, sample_rate
            )
        else:
            # Fallback: only support PCM to WAV conversion
            if source_format == "pcm" and target_format == "wav":
                return self._pcm_to_wav(audio_data, sample_rate or 24000)
            else:
                raise RuntimeError(
                    f"Cannot convert from {source_format} to {target_format}. "
                    "Please install pydub or soundfile for audio conversion."
                )
    
    def _numpy_to_bytes(self, audio_array: 'np.ndarray') -> bytes:
        """Convert numpy array to bytes (16-bit PCM)"""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for array conversion but is not installed")
        
        # Ensure array is in the correct format
        if audio_array.dtype != np.int16:
            # Assume float32 in range [-1, 1]
            if audio_array.dtype in [np.float32, np.float64]:
                audio_array = (audio_array * 32767).astype(np.int16)
            else:
                audio_array = audio_array.astype(np.int16)
        
        return audio_array.tobytes()
    
    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """Convert raw PCM to WAV format with header"""
        # WAV header for 16-bit mono PCM
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        
        # Create WAV header
        header = bytearray()
        header.extend(b'RIFF')
        header.extend((36 + len(pcm_data)).to_bytes(4, 'little'))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend((16).to_bytes(4, 'little'))  # fmt chunk size
        header.extend((1).to_bytes(2, 'little'))   # PCM format
        header.extend((channels).to_bytes(2, 'little'))
        header.extend((sample_rate).to_bytes(4, 'little'))
        header.extend((byte_rate).to_bytes(4, 'little'))
        header.extend((block_align).to_bytes(2, 'little'))
        header.extend((bits_per_sample).to_bytes(2, 'little'))
        header.extend(b'data')
        header.extend((len(pcm_data)).to_bytes(4, 'little'))
        
        return bytes(header) + pcm_data
    
    async def _convert_with_pydub(
        self,
        audio_data: bytes,
        target_format: str,
        source_format: Optional[str],
        sample_rate: Optional[int]
    ) -> bytes:
        """Convert audio using pydub"""
        try:
            # Load audio
            if source_format == "pcm":
                # Create WAV from PCM for pydub
                wav_data = self._pcm_to_wav(audio_data, sample_rate or 24000)
                audio = AudioSegment.from_wav(io.BytesIO(wav_data))
            elif source_format:
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format=source_format)
            else:
                # Let pydub detect format
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Set output parameters
            output_params = {
                "format": target_format,
            }
            
            # Add bitrate for lossy formats
            if target_format in self.BIT_RATES:
                output_params["bitrate"] = self.BIT_RATES[target_format]
            
            # Add sample rate if different
            target_sample_rate = sample_rate or self.SAMPLE_RATES.get(target_format, 44100)
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Export to target format
            output_buffer = io.BytesIO()
            
            # Special handling for M4B (audiobook format)
            if target_format == "m4b":
                # M4B is essentially M4A with different extension
                # Use AAC codec which is standard for M4B
                output_params["format"] = "mp4"
                output_params["codec"] = "aac"
                audio.export(output_buffer, **output_params)
            else:
                audio.export(output_buffer, **output_params)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Pydub conversion failed: {e}")
            raise RuntimeError(f"Audio conversion failed: {e}")
    
    async def _convert_with_soundfile(
        self,
        audio_data: bytes,
        target_format: str,
        source_format: Optional[str],
        sample_rate: Optional[int]
    ) -> bytes:
        """Convert audio using soundfile (limited format support)"""
        try:
            # Read audio data
            if source_format == "pcm":
                if not NUMPY_AVAILABLE:
                    raise RuntimeError("numpy is required for PCM conversion but is not installed")
                # Convert PCM bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32767.0
                sr = sample_rate or 24000
            else:
                # Read from bytes
                audio_float, sr = sf.read(io.BytesIO(audio_data))
            
            # Convert sample rate if needed
            target_sr = sample_rate or self.SAMPLE_RATES.get(target_format, 44100)
            if sr != target_sr:
                # Simple linear interpolation (not ideal but functional)
                import scipy.signal
                audio_float = scipy.signal.resample(
                    audio_float, 
                    int(len(audio_float) * target_sr / sr)
                )
            
            # Write to target format
            output_buffer = io.BytesIO()
            sf.write(output_buffer, audio_float, target_sr, format=target_format.upper())
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Soundfile conversion failed: {e}")
            raise RuntimeError(f"Audio conversion failed: {e}")
    
    def get_supported_formats(self) -> Dict[str, bool]:
        """Get dictionary of supported audio formats"""
        base_formats = {
            "pcm": True,  # Always supported
            "wav": True,  # Can always create from PCM
        }
        
        if PYDUB_AVAILABLE:
            # Pydub supports many formats via ffmpeg
            base_formats.update({
                "mp3": True,
                "opus": True,
                "aac": True,
                "flac": True,
                "ogg": True,
                "m4a": True,
                "m4b": True,  # M4B is supported via pydub (same as M4A)
            })
        elif SOUNDFILE_AVAILABLE:
            # Soundfile has limited format support
            base_formats.update({
                "flac": True,
                "ogg": True,
            })
        
        return base_formats
    
    async def create_m4b_with_chapters(
        self,
        audio_files: list,
        chapter_titles: list,
        output_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Create an M4B audiobook file with chapter markers.
        
        Args:
            audio_files: List of audio file paths
            chapter_titles: List of chapter titles
            output_path: Output M4B file path
            metadata: Optional metadata (title, author, etc.)
            
        Returns:
            Success status
            
        Note: This requires ffmpeg to be installed for chapter support
        """
        if not PYDUB_AVAILABLE:
            logger.error("pydub is required for M4B creation")
            return False
        
        try:
            # Combine all audio files
            combined = AudioSegment.empty()
            chapter_times = [0]  # Start time of each chapter in milliseconds
            
            for audio_file in audio_files:
                audio = AudioSegment.from_file(audio_file)
                combined += audio
                chapter_times.append(len(combined))
            
            # Create metadata file for chapters (FFmpeg format)
            metadata_content = ";FFMETADATA1\n"
            
            # Add general metadata
            if metadata:
                for key, value in metadata.items():
                    metadata_content += f"{key}={value}\n"
            
            # Add chapters
            for i, (title, start_time) in enumerate(zip(chapter_titles, chapter_times[:-1])):
                end_time = chapter_times[i + 1] if i + 1 < len(chapter_times) else len(combined)
                metadata_content += f"\n[CHAPTER]\n"
                metadata_content += f"TIMEBASE=1/1000\n"
                metadata_content += f"START={start_time}\n"
                metadata_content += f"END={end_time}\n"
                metadata_content += f"title={title}\n"
            
            # Write metadata to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(metadata_content)
                metadata_path = f.name
            
            # Export with metadata using ffmpeg
            try:
                # First export to temporary m4a
                temp_audio = tempfile.NamedTemporaryFile(suffix='.m4a', delete=False)
                combined.export(temp_audio.name, format="mp4", codec="aac", bitrate="128k")
                
                # Use ffmpeg to add metadata and create final M4B
                import subprocess
                cmd = [
                    'ffmpeg', '-i', temp_audio.name,
                    '-i', metadata_path,
                    '-map_metadata', '1',
                    '-c', 'copy',
                    '-f', 'mp4',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    # Fallback: just save as M4B without chapters
                    combined.export(output_path, format="mp4", codec="aac", bitrate="128k")
                
                # Cleanup
                os.unlink(temp_audio.name)
                
            finally:
                os.unlink(metadata_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create M4B with chapters: {e}")
            return False
    
    async def validate_audio(self, audio_data: bytes, expected_format: str) -> bool:
        """
        Validate that audio data is in the expected format.
        
        Args:
            audio_data: Audio data to validate
            expected_format: Expected audio format
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if expected_format == "pcm":
                # PCM is raw data, just check if we have bytes
                return len(audio_data) > 0 and len(audio_data) % 2 == 0
            
            elif expected_format == "wav":
                # Check WAV header
                return audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]
            
            elif PYDUB_AVAILABLE:
                # Try to load with pydub
                AudioSegment.from_file(io.BytesIO(audio_data), format=expected_format)
                return True
            
            else:
                # Can't validate without libraries
                logger.warning(f"Cannot validate {expected_format} format without pydub")
                return True  # Assume valid
                
        except Exception as e:
            logger.debug(f"Audio validation failed: {e}")
            return False


# Singleton instance
_audio_service_instance: Optional[AudioService] = None

def get_audio_service() -> AudioService:
    """Get the singleton AudioService instance"""
    global _audio_service_instance
    if _audio_service_instance is None:
        _audio_service_instance = AudioService()
    return _audio_service_instance

#
# End of audio_service.py
#######################################################################################################################