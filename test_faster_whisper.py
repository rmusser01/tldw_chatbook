#!/usr/bin/env python3
"""
Simple test script to verify faster-whisper is working correctly.
"""

import sys
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_faster_whisper():
    """Test basic faster-whisper functionality."""
    
    # Test 1: Check if faster-whisper can be imported
    logger.info("Test 1: Checking if faster-whisper can be imported...")
    try:
        from faster_whisper import WhisperModel
        logger.info("✓ faster-whisper imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import faster-whisper: {e}")
        logger.error("Please install with: pip install faster-whisper")
        return False
    
    # Test 2: Try to load a small model
    logger.info("\nTest 2: Loading a small Whisper model (tiny)...")
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {type(e).__name__}: {e}")
        return False
    
    # Test 3: Create a test audio file (silent) and try to transcribe it
    logger.info("\nTest 3: Testing transcription on a test audio file...")
    
    # First check if we have a test audio file
    test_files = [
        "/tmp/test_audio.wav",
        Path.home() / "test_audio.wav",
        Path(__file__).parent / "test_audio.wav"
    ]
    
    test_audio = None
    for test_file in test_files:
        if os.path.exists(test_file):
            test_audio = str(test_file)
            logger.info(f"Found test audio file: {test_audio}")
            break
    
    if not test_audio:
        logger.warning("No test audio file found. Creating a simple test with numpy...")
        try:
            import numpy as np
            import wave
            
            # Create a 1-second silent audio file
            sample_rate = 16000
            duration = 1.0
            samples = int(sample_rate * duration)
            audio_data = np.zeros(samples, dtype=np.int16)
            
            test_audio = "/tmp/test_audio_silent.wav"
            with wave.open(test_audio, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)   # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            logger.info(f"Created silent test audio file: {test_audio}")
        except Exception as e:
            logger.warning(f"Could not create test audio file: {e}")
            logger.info("Skipping transcription test")
            return True
    
    # Try to transcribe
    try:
        logger.info(f"Transcribing {test_audio}...")
        segments, info = model.transcribe(test_audio, language="en")
        
        # Collect segments
        transcribed_text = ""
        segment_count = 0
        for segment in segments:
            transcribed_text += segment.text
            segment_count += 1
            logger.info(f"Segment {segment_count}: '{segment.text.strip()}'")
        
        logger.info(f"✓ Transcription completed successfully")
        logger.info(f"  Total segments: {segment_count}")
        logger.info(f"  Language detected: {info.language if hasattr(info, 'language') else 'Unknown'}")
        logger.info(f"  Audio duration: {info.duration if hasattr(info, 'duration') else 'Unknown'}s")
        
        # Clean up test file if we created it
        if "/tmp/test_audio_silent.wav" in test_audio:
            os.unlink(test_audio)
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Transcription failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("Testing faster-whisper installation and functionality...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    success = test_faster_whisper()
    
    if success:
        logger.info("\n✓ All tests passed! faster-whisper is working correctly.")
    else:
        logger.error("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()