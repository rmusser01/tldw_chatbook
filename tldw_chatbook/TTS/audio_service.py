# audio_service.py
# Description: Audio processing and format conversion service for TTS
#
# Imports
import io
import asyncio
from typing import AsyncGenerator, Optional, Union, BinaryIO
import numpy as np
from loguru import logger

# Third-party imports
try:
    import pyav
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    logger.warning("pyav not available. Some audio format conversions will be limited.")

try:
    import scipy.io.wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. WAV format support will be limited.")

#######################################################################################################################
#
# Classes and Functions:

class AudioFormat:
    """Supported audio formats"""
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"

class StreamingAudioWriter:
    """
    Handles streaming audio encoding for various formats.
    Inspired by the target app's StreamingAudioWriter pattern.
    """
    
    def __init__(self, format: str, sample_rate: int = 24000, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer = io.BytesIO()
        self.encoder = None
        self.container = None
        self.stream = None
        
        if self.format != AudioFormat.PCM and self.format != AudioFormat.WAV:
            if not PYAV_AVAILABLE:
                raise ImportError(f"pyav is required for {format} encoding. Install with: pip install av")
            self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize the audio encoder for the specified format"""
        if self.format == AudioFormat.MP3:
            codec_name = 'mp3'
            self.container = pyav.open(self.buffer, mode='w', format='mp3')
        elif self.format == AudioFormat.OPUS:
            codec_name = 'libopus'
            self.container = pyav.open(self.buffer, mode='w', format='ogg')
        elif self.format == AudioFormat.AAC:
            codec_name = 'aac'
            self.container = pyav.open(self.buffer, mode='w', format='adts')
        elif self.format == AudioFormat.FLAC:
            codec_name = 'flac'
            self.container = pyav.open(self.buffer, mode='w', format='flac')
        else:
            raise ValueError(f"Unsupported audio format: {self.format}")
        
        self.stream = self.container.add_stream(codec_name, rate=self.sample_rate)
        self.stream.channels = self.channels
    
    def write_chunk(self, audio_data: np.ndarray, is_last: bool = False) -> bytes:
        """
        Write audio data chunk and return encoded bytes if available.
        
        Args:
            audio_data: Audio samples as numpy array (float32 or int16)
            is_last: Whether this is the last chunk
            
        Returns:
            Encoded audio bytes (may be empty for some codecs until enough data is buffered)
        """
        output_bytes = b""
        
        if self.format == AudioFormat.PCM:
            # PCM: Just convert to int16 and return raw bytes
            if audio_data.dtype == np.float32:
                audio_data = np.int16(audio_data * 32767)
            return audio_data.tobytes()
        
        elif self.format == AudioFormat.WAV:
            # WAV: We need to collect all chunks and write at once (can't stream WAV properly)
            # This is a limitation of the WAV format which needs headers
            logger.warning("WAV format doesn't support true streaming. Collecting chunks...")
            return b""  # Return empty until finalize
        
        else:
            # Use pyav for other formats
            if audio_data.dtype == np.float32:
                audio_data = np.int16(audio_data * 32767)
            
            # Create audio frame
            frame = pyav.AudioFrame.from_ndarray(audio_data.reshape(1, -1), format='s16', layout='mono')
            frame.sample_rate = self.sample_rate
            
            # Encode frame
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            
            # Get any available output
            current_pos = self.buffer.tell()
            self.buffer.seek(0)
            output_bytes = self.buffer.read()
            self.buffer.seek(0)
            self.buffer.truncate(0)
            self.buffer.seek(0)
        
        if is_last:
            output_bytes += self.finalize()
        
        return output_bytes
    
    def finalize(self) -> bytes:
        """Finalize encoding and return any remaining bytes"""
        final_bytes = b""
        
        if self.format == AudioFormat.WAV:
            # For WAV, we need to write the complete file with headers
            # This is a simplified implementation
            logger.warning("WAV finalization not fully implemented in streaming mode")
            return b""
        
        elif self.container:
            # Flush encoder
            for packet in self.stream.encode(None):
                self.container.mux(packet)
            
            # Close container
            self.container.close()
            
            # Get final output
            self.buffer.seek(0)
            final_bytes = self.buffer.read()
            self.buffer.close()
        
        return final_bytes

class AudioService:
    """
    Main audio processing service for TTS.
    Handles format conversion and audio processing.
    """
    
    def __init__(self):
        self.logger = logger.bind(service="AudioService")
    
    async def convert_audio_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        output_format: str,
        sample_rate: int = 24000,
        channels: int = 1
    ) -> AsyncGenerator[bytes, None]:
        """
        Convert streaming audio data to the specified format.
        
        Args:
            audio_stream: Async generator yielding numpy arrays of audio data
            output_format: Target audio format
            sample_rate: Sample rate of the audio
            channels: Number of audio channels
            
        Yields:
            Encoded audio chunks in the target format
        """
        writer = StreamingAudioWriter(output_format, sample_rate, channels)
        
        try:
            async for audio_chunk in audio_stream:
                encoded_chunk = writer.write_chunk(audio_chunk, is_last=False)
                if encoded_chunk:
                    yield encoded_chunk
            
            # Finalize and yield any remaining data
            final_chunk = writer.finalize()
            if final_chunk:
                yield final_chunk
                
        except Exception as e:
            self.logger.error(f"Error in audio conversion: {e}")
            raise
    
    def convert_audio_sync(
        self,
        audio_data: np.ndarray,
        output_format: str,
        sample_rate: int = 24000,
        channels: int = 1
    ) -> bytes:
        """
        Synchronously convert audio data to the specified format.
        
        Args:
            audio_data: Audio samples as numpy array
            output_format: Target audio format
            sample_rate: Sample rate of the audio
            channels: Number of audio channels
            
        Returns:
            Complete encoded audio file as bytes
        """
        if output_format == AudioFormat.WAV and SCIPY_AVAILABLE:
            # Use scipy for simple WAV conversion
            buffer = io.BytesIO()
            if audio_data.dtype == np.float32:
                audio_data = np.int16(audio_data * 32767)
            scipy.io.wavfile.write(buffer, sample_rate, audio_data)
            buffer.seek(0)
            return buffer.read()
        
        elif output_format == AudioFormat.PCM:
            # Simple PCM conversion
            if audio_data.dtype == np.float32:
                audio_data = np.int16(audio_data * 32767)
            return audio_data.tobytes()
        
        else:
            # Use streaming writer for other formats
            writer = StreamingAudioWriter(output_format, sample_rate, channels)
            output = writer.write_chunk(audio_data, is_last=True)
            return output

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