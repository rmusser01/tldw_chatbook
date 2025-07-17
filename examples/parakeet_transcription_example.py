#!/usr/bin/env python3
"""
Example script demonstrating NVIDIA Parakeet transcription usage in tldw_chatbook.

This example shows how to use the new Parakeet TDT model for audio transcription,
which offers lower latency and better streaming performance compared to Whisper models.

Requirements:
    pip install -e ".[audio]"
    
    Note: NeMo toolkit will be installed as part of the audio dependencies.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService
from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor


def example_direct_transcription():
    """Example of direct transcription using TranscriptionService."""
    print("=== Direct Transcription Example ===\n")
    
    # Initialize the transcription service
    service = TranscriptionService()
    
    # List available models
    print("Available transcription models:")
    models = service.list_available_models()
    for provider, model_list in models.items():
        print(f"\n{provider}:")
        for model in model_list:
            print(f"  - {model}")
    
    # Example audio file path (replace with your own audio file)
    audio_file = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_file):
        print(f"\nPlease update the audio_file path to point to a real audio file.")
        return
    
    print(f"\nTranscribing: {audio_file}")
    
    try:
        # Transcribe with Parakeet TDT (optimized for real-time)
        print("\nUsing Parakeet TDT model...")
        result = service.transcribe(
            audio_path=audio_file,
            provider="parakeet",
            model="nvidia/parakeet-tdt-1.1b",  # TDT = Transducer, best for streaming
            language="en"
        )
        
        print(f"Transcription: {result['text']}")
        print(f"Provider: {result.get('provider', 'unknown')}")
        print(f"Model: {result.get('model', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_audio_processing():
    """Example of audio processing with Parakeet through LocalAudioProcessor."""
    print("\n\n=== Audio Processing Example ===\n")
    
    # Initialize the audio processor
    processor = LocalAudioProcessor()
    
    # Process audio files with Parakeet
    audio_urls = [
        "https://example.com/audio1.mp3",  # Replace with real URLs
        "path/to/local/audio2.wav"
    ]
    
    print("Processing audio files with Parakeet transcription...")
    
    try:
        results = processor.process_audio_files(
            inputs=audio_urls,
            transcription_model="nvidia/parakeet-tdt-1.1b",  # Use Parakeet TDT
            transcription_language="en",
            perform_chunking=True,
            chunk_method="sentences",
            max_chunk_size=500,
            perform_analysis=True,
            api_name="openai",  # For summarization
            custom_prompt="Summarize the key points from this transcription.",
            keep_original=True
        )
        
        print(f"\nProcessed: {results['processed_count']} files")
        print(f"Errors: {results['errors_count']}")
        
        for result in results['results']:
            if result['status'] == 'Success':
                print(f"\nFile: {result['input_ref']}")
                print(f"Transcription length: {len(result.get('content', ''))} characters")
                if result.get('analysis'):
                    print(f"Summary: {result['analysis'][:200]}...")
            else:
                print(f"\nError processing {result['input_ref']}: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"Error: {e}")


def example_model_comparison():
    """Example comparing different Parakeet models."""
    print("\n\n=== Model Comparison Example ===\n")
    
    service = TranscriptionService()
    audio_file = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_file):
        print("Please update the audio_file path to point to a real audio file.")
        return
    
    # Different Parakeet models for different use cases
    models = [
        ("nvidia/parakeet-tdt-1.1b", "Transducer - Best for streaming/real-time"),
        ("nvidia/parakeet-ctc-1.1b", "CTC - Fast, good for batch processing"),
        ("nvidia/parakeet-rnnt-1.1b", "RNN-T - Balance of accuracy and speed")
    ]
    
    for model_name, description in models:
        print(f"\nTesting {model_name} ({description})...")
        try:
            result = service.transcribe(
                audio_path=audio_file,
                provider="parakeet",
                model=model_name
            )
            print(f"Result: {result['text'][:100]}...")
        except Exception as e:
            print(f"Error with {model_name}: {e}")


def example_configuration():
    """Example showing how to configure Parakeet in config.toml."""
    print("\n\n=== Configuration Example ===\n")
    
    config_example = """
# In your ~/.config/tldw_cli/config.toml file:

[transcription]
# Use Parakeet as the default provider for lower latency
default_provider = "parakeet"

# Default to the TDT model for best streaming performance
default_model = "nvidia/parakeet-tdt-1.1b"

# Language for transcription
default_language = "en"

# Use GPU if available
device = "cuda"  # or "cpu" if no GPU

# Voice Activity Detection
use_vad_by_default = true  # Recommended for Parakeet
"""
    
    print("To use Parakeet as your default transcription provider:")
    print(config_example)


if __name__ == "__main__":
    print("NVIDIA Parakeet Transcription Examples\n")
    print("=======================================\n")
    
    # Check if NeMo is available
    try:
        import nemo
        print("✓ NeMo toolkit is installed\n")
    except ImportError:
        print("✗ NeMo toolkit is not installed")
        print("  Install with: pip install -e \".[audio]\"\n")
        sys.exit(1)
    
    # Run examples
    example_direct_transcription()
    example_configuration()
    
    # Uncomment these to run if you have audio files:
    # example_audio_processing()
    # example_model_comparison()