#!/usr/bin/env python3
"""
Example script demonstrating speaker diarization with transcription.

This example shows how to transcribe audio with speaker identification,
identifying who said what in a conversation or meeting.

Requirements:
    pip install -e ".[diarization]"
    
    Note: This installs additional dependencies:
    - torch (PyTorch)
    - speechbrain
    - silero-vad
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService


def example_basic_diarization():
    """Basic example of transcription with speaker diarization."""
    print("=== Basic Diarization Example ===\n")
    
    # Initialize the transcription service
    service = TranscriptionService()
    
    # Check if diarization is available
    if not service.is_diarization_available():
        print("❌ Speaker diarization is not available.")
        print("\nMissing dependencies:")
        requirements = service.get_diarization_requirements()
        for dep, available in requirements.items():
            status = "✓" if available else "✗"
            print(f"  {status} {dep}")
        print("\nInstall with: pip install -e \".[diarization]\"")
        return
    
    print("✓ Speaker diarization is available\n")
    
    # Example audio file path (replace with your own audio file)
    audio_file = "path/to/your/conversation.wav"
    
    if not os.path.exists(audio_file):
        print(f"Please update the audio_file path to point to a real audio file.")
        print(f"Current path: {audio_file}")
        return
    
    print(f"Transcribing with diarization: {audio_file}")
    
    try:
        # Transcribe with speaker diarization
        result = service.transcribe(
            audio_path=audio_file,
            provider="faster-whisper",
            model="base",
            language="en",
            diarize=True,  # Enable speaker diarization
            progress_callback=lambda p, s, d: print(f"Progress: {p:.0f}% - {s}")
        )
        
        # Check if diarization was performed
        if result.get('diarization_performed', False):
            print(f"\n✓ Diarization successful!")
            print(f"Number of speakers detected: {result['num_speakers']}")
            
            # Print speaker statistics
            if 'speakers' in result:
                print("\nSpeaker Statistics:")
                for speaker in result['speakers']:
                    print(f"  {speaker['speaker_label']}: {speaker['total_time']:.1f}s speaking time")
            
            # Print transcription with speakers
            print("\nTranscription with speaker labels:")
            print("-" * 50)
            
            for segment in result['segments']:
                speaker = segment.get('speaker_label', 'UNKNOWN')
                text = segment.get('text', '')
                print(f"[{speaker}] {text}")
            
        else:
            print("\n❌ Diarization was not performed")
            if 'diarization_error' in result:
                print(f"Error: {result['diarization_error']}")
            
            # Show regular transcription
            print("\nRegular transcription (no speaker labels):")
            print(result['text'])
            
    except Exception as e:
        print(f"Error: {e}")


def example_advanced_diarization():
    """Advanced example with custom diarization settings."""
    print("\n\n=== Advanced Diarization Example ===\n")
    
    service = TranscriptionService()
    
    if not service.is_diarization_available():
        print("Diarization not available. See basic example for details.")
        return
    
    audio_file = "path/to/your/meeting.wav"
    
    if not os.path.exists(audio_file):
        print(f"Please update the audio_file path to point to a real audio file.")
        return
    
    print(f"Transcribing with custom diarization settings: {audio_file}")
    
    try:
        # Transcribe with custom settings
        result = service.transcribe(
            audio_path=audio_file,
            provider="faster-whisper",
            diarize=True,
            num_speakers=3,  # Specify expected number of speakers if known
            diarization_config={
                'clustering_method': 'spectral',  # or 'agglomerative'
                'min_speakers': 2,
                'max_speakers': 5,
                'merge_threshold': 0.5  # Seconds between segments to merge
            }
        )
        
        if result.get('diarization_performed'):
            # Format and save results
            formatted = service.format_segments_with_timestamps(
                result['segments'],
                include_timestamps=True,
                include_speakers=True
            )
            
            print("\nFormatted transcription:")
            print(formatted)
            
            # Save to file
            output_file = audio_file.replace('.wav', '_diarized.txt')
            with open(output_file, 'w') as f:
                f.write(f"Audio: {audio_file}\n")
                f.write(f"Speakers: {result['num_speakers']}\n")
                f.write("-" * 50 + "\n")
                f.write(formatted)
            
            print(f"\nResults saved to: {output_file}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_export_formats():
    """Example of exporting diarized transcription in different formats."""
    print("\n\n=== Export Formats Example ===\n")
    
    # Mock diarization result for demonstration
    result = {
        'text': 'Full transcription text here...',
        'num_speakers': 2,
        'diarization_performed': True,
        'segments': [
            {
                'start': 0.0, 'end': 2.5,
                'text': 'Hello, welcome to the meeting.',
                'speaker_id': 0, 'speaker_label': 'SPEAKER_0'
            },
            {
                'start': 3.0, 'end': 5.5,
                'text': 'Thank you for having me.',
                'speaker_id': 1, 'speaker_label': 'SPEAKER_1'
            },
            {
                'start': 6.0, 'end': 9.0,
                'text': "Let's discuss the agenda.",
                'speaker_id': 0, 'speaker_label': 'SPEAKER_0'
            }
        ],
        'speakers': [
            {'speaker_id': 0, 'speaker_label': 'SPEAKER_0', 'total_time': 5.5},
            {'speaker_id': 1, 'speaker_label': 'SPEAKER_1', 'total_time': 2.5}
        ]
    }
    
    # Export as JSON
    import json
    json_output = json.dumps(result, indent=2)
    print("JSON Format:")
    print(json_output[:200] + "...")
    
    # Export as SRT (subtitle format)
    print("\n\nSRT Format:")
    for i, segment in enumerate(result['segments'], 1):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        speaker = segment['speaker_label']
        text = segment['text']
        
        print(f"{i}")
        print(f"{start_time} --> {end_time}")
        print(f"[{speaker}] {text}")
        print()
    
    # Export as simple dialogue
    print("\nDialogue Format:")
    for segment in result['segments']:
        speaker = segment['speaker_label']
        text = segment['text']
        print(f"{speaker}: {text}")


def format_srt_time(seconds):
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def example_configuration():
    """Example showing diarization configuration in config.toml."""
    print("\n\n=== Configuration Example ===\n")
    
    config_example = """
# In your ~/.config/tldw_cli/config.toml file:

[transcription]
# Enable diarization by default
use_diarization_by_default = true

[diarization]
# Enable/disable diarization
enabled = true

# VAD settings
vad_threshold = 0.5  # Sensitivity (0.0-1.0)
vad_min_speech_duration = 0.25
vad_min_silence_duration = 0.25

# Segmentation
segment_duration = 2.0  # Analysis window
segment_overlap = 0.5

# Speaker embedding
embedding_model = "speechbrain/spkrec-ecapa-voxceleb"
embedding_device = "auto"  # auto, cuda, cpu

# Clustering
clustering_method = "spectral"
min_speakers = 1
max_speakers = 10

# Post-processing
merge_threshold = 0.5  # Gap to merge segments
min_speaker_duration = 3.0  # Min total time per speaker
"""
    
    print("To configure diarization defaults:")
    print(config_example)
    
    print("\nCurrent configuration:")
    service = TranscriptionService()
    print(f"Diarization available: {service.is_diarization_available()}")


if __name__ == "__main__":
    print("Speaker Diarization Examples")
    print("============================\n")
    
    # Check basic availability
    service = TranscriptionService()
    requirements = service.get_diarization_requirements()
    
    print("Dependency Status:")
    all_available = True
    for dep, available in requirements.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
        if not available:
            all_available = False
    
    if not all_available:
        print("\nSome dependencies are missing.")
        print("Install with: pip install -e \".[diarization]\"")
        print("\nNote: This will install PyTorch, which can be large (>1GB)")
        print("For GPU support, see: https://pytorch.org/get-started/locally/")
    
    print("\n" + "=" * 50 + "\n")
    
    # Run examples
    example_basic_diarization()
    example_configuration()
    example_export_formats()
    
    # Uncomment to run advanced example with real audio
    # example_advanced_diarization()