#!/usr/bin/env python3
"""Debug script to test transcription service directly."""

import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService, TranscriptionError
    print("✓ Successfully imported TranscriptionService")
except Exception as e:
    print(f"✗ Failed to import TranscriptionService: {e}")
    sys.exit(1)

# Test basic instantiation
try:
    # Mock the config
    with patch('tldw_chatbook.Local_Ingestion.transcription_service.get_cli_setting') as mock_get_setting:
        def get_setting_side_effect(key, default=None):
            settings = {
                'transcription.provider': 'faster-whisper',
                'transcription.model': 'base',
                'transcription.language': 'en',
                'transcription.vad_filter': True,
                'transcription.compute_type': 'int8',
                'transcription.device': 'cpu',
                'transcription.device_index': 0,
                'transcription.num_workers': 1,
                'transcription.download_root': None,
                'transcription.local_files_only': False,
            }
            return settings.get(key, default)
        
        mock_get_setting.side_effect = get_setting_side_effect
        service = TranscriptionService()
        print("✓ Successfully created TranscriptionService instance")
        
        # Test the internal method call signature
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.FASTER_WHISPER_AVAILABLE', False):
            try:
                # This should raise TranscriptionError
                result = service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="en",
                    vad_filter=True
                )
                print("✗ Expected TranscriptionError but none was raised")
            except TranscriptionError as e:
                if "faster-whisper is not installed" in str(e):
                    print("✓ TranscriptionError raised correctly with expected message")
                else:
                    print(f"✗ TranscriptionError raised but with unexpected message: {e}")
            except TypeError as e:
                print(f"✗ TypeError raised (signature mismatch): {e}")
            except Exception as e:
                print(f"✗ Unexpected error: {type(e).__name__}: {e}")
                
except Exception as e:
    print(f"✗ Error during test: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()