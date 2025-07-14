#!/usr/bin/env python3
"""
Test script for Lightning Whisper MLX integration in tldw_chatbook.

This script tests the new lightning-whisper-mlx transcription backend.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import tldw_chatbook
sys.path.insert(0, str(Path(__file__).parent))

from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE


def test_lightning_whisper_mlx():
    """Test the Lightning Whisper MLX integration."""
    print("=" * 60)
    print("Testing Lightning Whisper MLX Integration")
    print("=" * 60)
    
    # Check platform
    print(f"\nPlatform: {sys.platform}")
    if sys.platform != 'darwin':
        print("❌ Lightning Whisper MLX is only available on macOS (darwin)")
        return
    
    # Check if lightning-whisper-mlx is available
    print(f"\nChecking dependencies...")
    if DEPENDENCIES_AVAILABLE.get('lightning_whisper_mlx', False):
        print("✅ lightning-whisper-mlx is available")
    else:
        print("❌ lightning-whisper-mlx is not installed")
        print("   Install with: pip install lightning-whisper-mlx")
        return
    
    # Initialize transcription service
    print("\nInitializing TranscriptionService...")
    service = TranscriptionService()
    
    # Check available providers
    providers = service.get_available_providers()
    print(f"\nAvailable providers: {providers}")
    
    if 'lightning-whisper-mlx' not in providers:
        print("❌ lightning-whisper-mlx not in available providers")
        return
    
    print("✅ lightning-whisper-mlx is available as a provider")
    
    # Check available models
    models = service.list_available_models('lightning-whisper-mlx')
    print(f"\nAvailable models for lightning-whisper-mlx:")
    for model in models.get('lightning-whisper-mlx', []):
        print(f"  - {model}")
    
    # Check default provider
    print(f"\nDefault provider: {service.config['default_provider']}")
    if service.config['default_provider'] == 'lightning-whisper-mlx':
        print("✅ lightning-whisper-mlx is set as the default provider on macOS")
    
    # Test with a sample audio file if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"\nTesting transcription with: {audio_file}")
            print("Using model: distil-medium.en")
            
            try:
                # Define progress callback
                def progress_callback(progress, status, data):
                    print(f"Progress: {progress}% - {status}")
                
                # Transcribe
                result = service.transcribe(
                    audio_file,
                    provider='lightning-whisper-mlx',
                    model='distil-medium.en',
                    source_lang='en',
                    progress_callback=progress_callback
                )
                
                print("\n✅ Transcription successful!")
                print(f"Provider: {result.get('provider')}")
                print(f"Model: {result.get('model')}")
                print(f"Batch size: {result.get('batch_size')}")
                print(f"Quantization: {result.get('quantization')}")
                print(f"Language: {result.get('language')}")
                print(f"Text length: {len(result.get('text', ''))}")
                print(f"\nFirst 200 characters:")
                print(result.get('text', '')[:200] + "...")
                
            except Exception as e:
                print(f"\n❌ Transcription failed: {str(e)}")
        else:
            print(f"\n❌ Audio file not found: {audio_file}")
    else:
        print("\n💡 Tip: Provide an audio file path to test transcription")
        print("   Usage: python test_lightning_whisper.py <audio_file>")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_lightning_whisper_mlx()