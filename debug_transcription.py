#!/usr/bin/env python3
"""
Debug script to test faster-whisper transcription with enhanced logging and timeout.
"""

import sys
import os
import time
import logging
import threading
import signal
from pathlib import Path

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable progress bars that might interfere
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TQDM_DISABLE'] = '1'

# Fix for macOS file descriptor issues
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    logger.error("Timeout reached! Process appears to be hanging.")
    raise TimeoutError("Transcription process timed out")

def test_faster_whisper_with_timeout(audio_file=None, timeout_seconds=300):
    """Test faster-whisper with a timeout mechanism."""
    
    logger.info(f"Starting faster-whisper test with {timeout_seconds}s timeout")
    
    # Set up timeout signal (Unix only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        # Test 1: Import faster-whisper
        logger.info("Test 1: Importing faster-whisper...")
        import_start = time.time()
        try:
            from faster_whisper import WhisperModel
            import_time = time.time() - import_start
            logger.info(f"✓ faster-whisper imported successfully in {import_time:.2f}s")
        except ImportError as e:
            logger.error(f"✗ Failed to import faster-whisper: {e}")
            return False
        
        # Test 2: Load model with progress updates
        logger.info("\nTest 2: Loading Whisper model...")
        model_load_start = time.time()
        
        # Create a thread to show progress
        stop_progress = threading.Event()
        def show_progress():
            """Show progress dots while loading."""
            dots = 0
            while not stop_progress.is_set():
                dots = (dots + 1) % 4
                print(f"\rLoading model{'.' * dots}{' ' * (3-dots)}", end='', flush=True)
                time.sleep(0.5)
            print("\r" + " " * 20 + "\r", end='', flush=True)  # Clear the line
        
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.start()
        
        try:
            logger.info("Creating WhisperModel instance...")
            logger.info("Model: tiny, Device: cpu, Compute type: int8")
            
            # Add file descriptor protection
            import subprocess
            original_popen = subprocess.Popen
            
            def safe_popen(*args, **kwargs):
                """Wrapper to ensure proper file descriptor handling."""
                # Force close_fds=True for safety
                kwargs['close_fds'] = True
                return original_popen(*args, **kwargs)
            
            subprocess.Popen = safe_popen
            
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            
            # Restore original Popen
            subprocess.Popen = original_popen
            
            stop_progress.set()
            progress_thread.join()
            
            model_load_time = time.time() - model_load_start
            logger.info(f"✓ Model loaded successfully in {model_load_time:.2f}s")
            
        except Exception as e:
            stop_progress.set()
            progress_thread.join()
            logger.error(f"✗ Failed to load model: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
            return False
        
        # Test 3: Transcribe audio
        if audio_file and os.path.exists(audio_file):
            logger.info(f"\nTest 3: Transcribing audio file: {audio_file}")
            transcribe_start = time.time()
            
            try:
                logger.info("Starting transcription...")
                segments, info = model.transcribe(audio_file, language="en")
                
                logger.info(f"Language: {info.language}, Probability: {info.language_probability:.2f}")
                
                # Process segments
                text_parts = []
                segment_count = 0
                for segment in segments:
                    text_parts.append(segment.text.strip())
                    segment_count += 1
                    if segment_count <= 3:  # Show first 3 segments
                        logger.info(f"Segment {segment_count}: [{segment.start:.1f}s - {segment.end:.1f}s] {segment.text.strip()[:50]}...")
                
                transcribe_time = time.time() - transcribe_start
                full_text = " ".join(text_parts)
                
                logger.info(f"✓ Transcription completed in {transcribe_time:.2f}s")
                logger.info(f"  Total segments: {segment_count}")
                logger.info(f"  Total characters: {len(full_text)}")
                logger.info(f"  First 200 chars: {full_text[:200]}...")
                
            except Exception as e:
                logger.error(f"✗ Transcription failed: {type(e).__name__}: {e}")
                logger.exception("Full traceback:")
                return False
        else:
            logger.info("\nTest 3: No audio file provided or file not found, skipping transcription test")
        
        return True
        
    except TimeoutError:
        logger.error("Process timed out! This suggests the model loading or transcription is hanging.")
        logger.error("Possible solutions:")
        logger.error("1. Pre-download the model: huggingface-cli download openai/whisper-tiny")
        logger.error("2. Check disk space and permissions in the cache directory")
        logger.error("3. Try running with: TRANSFORMERS_OFFLINE=1 python debug_transcription.py")
        return False
        
    finally:
        # Cancel timeout if set
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

def main():
    """Run the debug test."""
    logger.info("=== Faster-Whisper Debug Test ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Check command line arguments
    audio_file = None
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            audio_file = None
    
    # Run test with 5 minute timeout
    success = test_faster_whisper_with_timeout(audio_file, timeout_seconds=300)
    
    if success:
        logger.info("\n✓ All tests passed! faster-whisper is working correctly.")
    else:
        logger.error("\n✗ Tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()