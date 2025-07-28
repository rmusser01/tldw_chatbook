#!/usr/bin/env python3
"""
Test script to download a YouTube video and transcribe it using faster-whisper.
"""

import sys
import os
import time
import logging
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the video processor
from tldw_chatbook.Local_Ingestion.video_processing import LocalVideoProcessor

def test_youtube_transcription(url):
    """Test downloading and transcribing a YouTube video."""
    
    logger.info(f"Testing YouTube transcription with URL: {url}")
    
    # Create processor
    processor = LocalVideoProcessor()
    
    # Create progress callback
    def progress_callback(progress, status, data=None):
        """Show transcription progress."""
        print(f"\rTranscription: {status} [{progress:.0f}%]", end='', flush=True)
        if data:
            logger.debug(f"Progress data: {data}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Using temporary directory: {temp_dir}")
            
            # Test 1: Download audio from YouTube
            logger.info("\nTest 1: Downloading audio from YouTube...")
            download_start = time.time()
            
            try:
                audio_path = processor.download_video(
                    url=url,
                    output_dir=temp_dir,
                    download_video_flag=False  # Audio only
                )
                
                download_time = time.time() - download_start
                logger.info(f"✓ Audio downloaded successfully in {download_time:.2f}s")
                logger.info(f"  Audio file: {audio_path}")
                logger.info(f"  File size: {os.path.getsize(audio_path) / (1024*1024):.2f} MB")
                
            except Exception as e:
                logger.error(f"✗ Download failed: {type(e).__name__}: {e}")
                return False
            
            # Test 2: Process the video (which includes transcription)
            logger.info("\nTest 2: Processing video with transcription...")
            process_start = time.time()
            
            try:
                result = processor.process_videos(
                    inputs=[url],
                    download_video_flag=False,
                    transcription_provider="faster-whisper",
                    transcription_model="tiny",
                    transcription_language="en",
                    transcription_progress_callback=progress_callback,
                    perform_chunking=False,
                    perform_analysis=False
                )
                
                print()  # New line after progress
                process_time = time.time() - process_start
                
                if result.get("processed_count", 0) > 0:
                    logger.info(f"✓ Processing completed successfully in {process_time:.2f}s")
                    
                    # Show results
                    for item in result.get("results", []):
                        if item.get("status") == "Success":
                            logger.info(f"\nTranscription Results:")
                            logger.info(f"  Title: {item.get('metadata', {}).get('title', 'Unknown')}")
                            logger.info(f"  Duration: {item.get('metadata', {}).get('duration', 0):.1f}s")
                            logger.info(f"  Segments: {len(item.get('segments', []))}")
                            logger.info(f"  Characters: {len(item.get('content', ''))}")
                            
                            # Show first 500 characters
                            content = item.get('content', '')
                            if content:
                                logger.info(f"\nFirst 500 characters of transcription:")
                                logger.info(f"  {content[:500]}...")
                        else:
                            logger.error(f"✗ Processing failed: {item.get('error', 'Unknown error')}")
                else:
                    logger.error(f"✗ No files were processed successfully")
                    if result.get("errors"):
                        for error in result["errors"]:
                            logger.error(f"  Error: {error}")
                
            except Exception as e:
                logger.error(f"✗ Processing failed: {type(e).__name__}: {e}")
                logger.exception("Full traceback:")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    """Run the test."""
    # Default test URL
    test_url = "https://www.youtube.com/watch?v=xg4cXaS5Wvg"
    
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    
    logger.info("=== YouTube Transcription Test ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Test URL: {test_url}")
    
    # Check if yt-dlp is available
    try:
        import yt_dlp
        logger.info("✓ yt-dlp is available")
    except ImportError:
        logger.error("✗ yt-dlp is not installed. Install with: pip install yt-dlp")
        sys.exit(1)
    
    # Check if faster-whisper is available
    try:
        import faster_whisper
        logger.info("✓ faster-whisper is available")
    except ImportError:
        logger.error("✗ faster-whisper is not installed. Install with: pip install faster-whisper")
        sys.exit(1)
    
    # Run the test
    success = test_youtube_transcription(test_url)
    
    if success:
        logger.info("\n✓ Test completed successfully!")
    else:
        logger.error("\n✗ Test failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()