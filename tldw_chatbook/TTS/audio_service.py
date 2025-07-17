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
    }
    
    BIT_RATES = {
        "mp3": "192k",
        "opus": "128k",
        "aac": "192k",
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
            })
        elif SOUNDFILE_AVAILABLE:
            # Soundfile has limited format support
            base_formats.update({
                "flac": True,
                "ogg": True,
            })
        
        return base_formats
    
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