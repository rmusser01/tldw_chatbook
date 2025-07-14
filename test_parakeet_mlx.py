#!/usr/bin/env python3
"""
Test script for Parakeet MLX integration in tldw_chatbook.

This script tests both file transcription and real-time STT capabilities.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np

# Add the parent directory to the path so we can import tldw_chatbook
sys.path.insert(0, str(Path(__file__).parent))

from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE


def test_parakeet_mlx():
    """Test the Parakeet MLX integration."""
    print("=" * 60)
    print("Testing Parakeet MLX Integration")
    print("=" * 60)
    
    # Check platform
    print(f"\nPlatform: {sys.platform}")
    if sys.platform != 'darwin':
        print("❌ Parakeet MLX is only available on macOS (darwin)")
        return
    
    # Check if parakeet-mlx is available
    print(f"\nChecking dependencies...")
    if DEPENDENCIES_AVAILABLE.get('parakeet_mlx', False):
        print("✅ parakeet-mlx is available")
    else:
        print("❌ parakeet-mlx is not installed")
        print("   Install with: pip install parakeet-mlx -U")
        return
    
    # Initialize transcription service
    print("\nInitializing TranscriptionService...")
    service = TranscriptionService()
    
    # Check available providers
    providers = service.get_available_providers()
    print(f"\nAvailable providers: {providers}")
    
    if 'parakeet-mlx' not in providers:
        print("❌ parakeet-mlx not in available providers")
        return
    
    print("✅ parakeet-mlx is available as a provider")
    
    # Check available models
    models = service.list_available_models('parakeet-mlx')
    print(f"\nAvailable models for parakeet-mlx:")
    for model in models.get('parakeet-mlx', []):
        print(f"  - {model}")
    
    # Check default provider
    print(f"\nDefault provider: {service.config['default_provider']}")
    if service.config['default_provider'] == 'parakeet-mlx':
        print("✅ parakeet-mlx is set as the default provider on macOS")
    
    # Test file transcription if audio file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"\n{'='*60}")
            print("Testing FILE TRANSCRIPTION")
            print(f"{'='*60}")
            print(f"Audio file: {audio_file}")
            
            try:
                # Define progress callback
                def progress_callback(progress, status, data):
                    print(f"Progress: {progress}% - {status}")
                
                # Transcribe
                start_time = time.time()
                result = service.transcribe(
                    audio_file,
                    provider='parakeet-mlx',
                    model='mlx-community/parakeet-tdt-0.6b-v2',
                    source_lang='en',
                    progress_callback=progress_callback
                )
                end_time = time.time()
                
                print("\n✅ File transcription successful!")
                print(f"Provider: {result.get('provider')}")
                print(f"Model: {result.get('model')}")
                print(f"Precision: {result.get('precision')}")
                print(f"Attention type: {result.get('attention_type')}")
                print(f"Language: {result.get('language')}")
                print(f"Text length: {len(result.get('text', ''))}")
                print(f"Segments: {len(result.get('segments', []))}")
                print(f"Processing time: {end_time - start_time:.2f} seconds")
                
                if result.get('duration'):
                    speed = result['duration'] / (end_time - start_time)
                    print(f"Speed: {speed:.2f}x realtime")
                
                print(f"\nFirst 200 characters:")
                print(result.get('text', '')[:200] + "...")
                
            except Exception as e:
                print(f"\n❌ File transcription failed: {str(e)}")
        else:
            print(f"\n❌ Audio file not found: {audio_file}")
    
    # Test streaming transcription
    print(f"\n{'='*60}")
    print("Testing STREAMING TRANSCRIPTION")
    print(f"{'='*60}")
    
    try:
        # Create streaming transcriber
        streamer = service.create_streaming_transcriber(
            provider='parakeet-mlx',
            model='mlx-community/parakeet-tdt-0.6b-v2'
        )
        
        if streamer:
            print("✅ Created streaming transcriber")
            
            # Simulate streaming audio (16kHz, 0.5 second chunks)
            sample_rate = 16000
            chunk_duration = 0.5
            chunk_size = int(sample_rate * chunk_duration)
            
            print("\nSimulating streaming audio...")
            print("(In a real application, this would come from a microphone)")
            
            # Generate some test audio chunks (silence for demo)
            for i in range(3):
                # Generate random noise as test audio
                audio_chunk = np.random.randn(chunk_size).astype(np.float32) * 0.01
                
                # Add audio to streamer
                result = streamer.add_audio(audio_chunk, sample_rate)
                
                if result:
                    print(f"\nChunk {i+1}: {result['text']}")
                    print(f"  Partial: {result['partial']}")
                    print(f"  Timestamp: {result['timestamp']}")
                else:
                    print(f"\nChunk {i+1}: No transcription yet")
                
                time.sleep(0.1)  # Simulate real-time delay
            
            # Finalize
            final_result = streamer.finalize()
            if final_result:
                print(f"\nFinal transcription: {final_result['text']}")
            
            print("\n✅ Streaming transcription test completed")
            
        else:
            print("❌ Failed to create streaming transcriber")
            
    except Exception as e:
        print(f"\n❌ Streaming transcription test failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    if len(sys.argv) <= 1:
        print("\n💡 Tip: Provide an audio file path to test file transcription")
        print("   Usage: python test_parakeet_mlx.py <audio_file>")


if __name__ == "__main__":
    test_parakeet_mlx()