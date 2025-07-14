#!/usr/bin/env python3
"""
Generate a test speech file using macOS say command.
"""

import subprocess
import os
import sys

def generate_speech():
    """Generate speech using macOS say command."""
    text = "Hello, this is a test of the parakeet MLX transcription system. The quick brown fox jumps over the lazy dog."
    output_file = "test_speech.aiff"
    
    try:
        # Use macOS say command to generate speech
        cmd = ['say', '-o', output_file, text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Generated speech file: {output_file}")
            print(f"   Text: '{text}'")
            
            # Convert to WAV format
            wav_file = "test_speech.wav"
            convert_cmd = ['afconvert', '-f', 'WAVE', '-d', 'LEI16', output_file, wav_file]
            subprocess.run(convert_cmd, capture_output=True)
            
            # Remove AIFF file
            os.remove(output_file)
            
            print(f"✅ Converted to WAV: {wav_file}")
            return wav_file
        else:
            print(f"❌ Failed to generate speech: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    audio_file = generate_speech()
    if audio_file:
        print(f"\nYou can now test transcription with: python test_parakeet_mlx.py {audio_file}")