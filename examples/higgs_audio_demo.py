#!/usr/bin/env python3
"""
Higgs Audio TTS Demo Script

This script demonstrates the basic usage of the Higgs Audio TTS backend
in tldw_chatbook, including voice cloning and multi-speaker dialog.

Requirements:
- tldw_chatbook with Higgs backend installed
- PyTorch and boson-multimodal
- A reference audio file for voice cloning (optional)
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager
from loguru import logger


async def basic_tts_demo():
    """Demonstrate basic text-to-speech generation"""
    print("\n=== Basic TTS Demo ===")
    
    # Initialize backend
    config = {
        "HIGGS_DEVICE": "auto",  # Auto-detect CUDA
        "HIGGS_TRACK_PERFORMANCE": True
    }
    
    manager = TTSBackendManager(config)
    backend = await manager.get_backend("local_higgs_v2")
    
    # Simple text generation
    text = "Hello! This is a demonstration of Higgs Audio text-to-speech. It can generate natural sounding speech in multiple languages."
    
    request = OpenAISpeechRequest(
        input=text,
        voice="professional_female",  # Use default voice
        response_format="wav"
    )
    
    print(f"Generating speech for: '{text[:50]}...'")
    
    # Generate and save audio
    audio_data = bytearray()
    async for chunk in backend.generate_speech_stream(request):
        audio_data.extend(chunk)
    
    # Save to file
    output_file = "higgs_demo_basic.wav"
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    print(f"✓ Saved audio to: {output_file}")
    
    # Get performance stats
    if hasattr(backend, 'get_performance_stats'):
        stats = backend.get_performance_stats()
        print(f"  Generation time: {stats.get('average_generation_time', 0):.2f}s")
    
    await manager.close_all_backends()


async def voice_cloning_demo(reference_audio_path: Optional[str] = None):
    """Demonstrate voice cloning capabilities"""
    print("\n=== Voice Cloning Demo ===")
    
    if not reference_audio_path:
        print("! No reference audio provided, skipping voice cloning demo")
        print("  Usage: python higgs_audio_demo.py /path/to/reference.wav")
        return
    
    # Check if reference exists
    ref_path = Path(reference_audio_path)
    if not ref_path.exists():
        print(f"! Reference audio not found: {reference_audio_path}")
        return
    
    # Initialize voice manager
    voice_dir = Path.home() / ".config" / "tldw_cli" / "higgs_voices"
    voice_manager = HiggsVoiceProfileManager(voice_dir)
    
    # Create voice profile
    profile_name = "demo_cloned_voice"
    print(f"Creating voice profile from: {ref_path.name}")
    
    success, message = voice_manager.create_profile(
        profile_name=profile_name,
        reference_audio_path=str(ref_path),
        display_name="Demo Cloned Voice",
        description="Voice cloned for demonstration",
        tags=["demo", "cloned"]
    )
    
    if not success:
        print(f"! Failed to create voice profile: {message}")
        return
    
    print(f"✓ Created voice profile: {profile_name}")
    
    # Use the cloned voice
    manager = TTSBackendManager({})
    backend = await manager.get_backend("local_higgs_v2")
    
    text = "This is my cloned voice speaking. The voice characteristics should match the reference audio you provided."
    
    request = OpenAISpeechRequest(
        input=text,
        voice=profile_name,  # Use the cloned voice
        response_format="wav"
    )
    
    print("Generating speech with cloned voice...")
    
    audio_data = bytearray()
    async for chunk in backend.generate_speech_stream(request):
        audio_data.extend(chunk)
    
    output_file = "higgs_demo_cloned.wav"
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    print(f"✓ Saved cloned voice audio to: {output_file}")
    
    # Cleanup (optional)
    # voice_manager.delete_profile(profile_name)
    
    await manager.close_all_backends()


async def multi_speaker_demo():
    """Demonstrate multi-speaker dialog generation"""
    print("\n=== Multi-Speaker Dialog Demo ===")
    
    # Initialize backend
    config = {
        "HIGGS_ENABLE_MULTI_SPEAKER": True,
        "HIGGS_SPEAKER_DELIMITER": "|||"
    }
    
    manager = TTSBackendManager(config)
    backend = await manager.get_backend("local_higgs_v2")
    
    # Multi-speaker dialog script
    dialog_script = """
Narrator|||In the bustling tech conference, two developers discuss the future of AI.
Alice|||Have you tried the new Higgs Audio model yet?
Bob|||Yes! The voice cloning is incredible. It only needs a few seconds of audio.
Alice|||I'm impressed by the multi-speaker support. Perfect for generating podcasts.
Bob|||Exactly! And it runs locally, no API costs or privacy concerns.
Narrator|||They continued exploring the possibilities, excited about the future.
"""
    
    print("Generating multi-speaker dialog...")
    print("Speakers: Narrator, Alice, Bob")
    
    request = OpenAISpeechRequest(
        input=dialog_script,
        voice="storyteller_male",  # Default narrator voice
        response_format="wav"
    )
    
    # Set progress callback
    async def progress_callback(info):
        if "current_chunk" in info:
            print(f"  Processing section {info['current_chunk']}/{info.get('total_chunks', '?')}")
    
    backend.set_progress_callback(progress_callback)
    
    # Generate dialog
    audio_data = bytearray()
    async for chunk in backend.generate_speech_stream(request):
        audio_data.extend(chunk)
    
    output_file = "higgs_demo_dialog.wav"
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    print(f"✓ Saved multi-speaker dialog to: {output_file}")
    
    await manager.close_all_backends()


async def list_voices_demo():
    """List available voice profiles"""
    print("\n=== Available Voices ===")
    
    # List default voices
    default_voices = [
        "professional_female",
        "warm_female", 
        "storyteller_male",
        "deep_male",
        "energetic_female",
        "soft_female"
    ]
    
    print("Default voices:")
    for voice in default_voices:
        print(f"  - {voice}")
    
    # List custom profiles
    voice_dir = Path.home() / ".config" / "tldw_cli" / "higgs_voices"
    voice_manager = HiggsVoiceProfileManager(voice_dir)
    
    profiles = voice_manager.list_profiles()
    if profiles:
        print("\nCustom voice profiles:")
        for profile in profiles:
            print(f"  - {profile['name']} ({profile['display_name']})")
            if profile['description']:
                print(f"    {profile['description']}")
    else:
        print("\nNo custom voice profiles found.")


async def progress_demo():
    """Demonstrate progress tracking during generation"""
    print("\n=== Progress Tracking Demo ===")
    
    # Initialize backend with progress tracking
    config = {"HIGGS_TRACK_PERFORMANCE": True}
    manager = TTSBackendManager(config)
    backend = await manager.get_backend("local_higgs_v2")
    
    # Longer text to show progress
    text = """
    Artificial intelligence has revolutionized many aspects of our daily lives. 
    From virtual assistants to autonomous vehicles, AI is everywhere. 
    One particularly impressive advancement is in speech synthesis, 
    where models like Higgs Audio can create incredibly natural sounding voices. 
    This technology opens up new possibilities for accessibility, 
    content creation, and human-computer interaction.
    """
    
    # Progress tracking callback
    async def detailed_progress(info):
        progress = info.get('progress', 0) * 100
        status = info.get('status', 'Processing')
        print(f"\r  {progress:5.1f}% - {status}", end='', flush=True)
    
    backend.set_progress_callback(detailed_progress)
    
    request = OpenAISpeechRequest(
        input=text,
        voice="professional_female",
        response_format="mp3"
    )
    
    print("Generating speech with progress tracking...")
    
    audio_data = bytearray()
    async for chunk in backend.generate_speech_stream(request):
        audio_data.extend(chunk)
    
    print()  # New line after progress
    
    output_file = "higgs_demo_progress.mp3"
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    print(f"✓ Saved audio to: {output_file}")
    
    # Show performance stats
    stats = backend.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Total generations: {stats.get('total_generations', 0)}")
    print(f"  Average time: {stats.get('average_generation_time', 0):.2f}s")
    print(f"  Total tokens: {stats.get('total_tokens', 0)}")
    
    await manager.close_all_backends()


async def main():
    """Run all demos"""
    print("Higgs Audio TTS Demo")
    print("=" * 50)
    
    # Check command line arguments
    reference_audio = None
    if len(sys.argv) > 1:
        reference_audio = sys.argv[1]
    
    try:
        # Run demos
        await basic_tts_demo()
        await multi_speaker_demo()
        await progress_demo()
        await list_voices_demo()
        
        # Voice cloning requires reference audio
        if reference_audio:
            await voice_cloning_demo(reference_audio)
        else:
            print("\n💡 Tip: Provide a reference audio file to test voice cloning:")
            print("   python higgs_audio_demo.py /path/to/reference.wav")
        
        print("\n✅ All demos completed successfully!")
        print("\nGenerated files:")
        print("  - higgs_demo_basic.wav")
        print("  - higgs_demo_dialog.wav") 
        print("  - higgs_demo_progress.mp3")
        if reference_audio:
            print("  - higgs_demo_cloned.wav")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have installed:")
        print("  - pip install torch boson-multimodal")
        print("  - pip install -e .[tts]  # From tldw_chatbook directory")


if __name__ == "__main__":
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="WARNING")  # Only show warnings and errors
    
    # Run async main
    asyncio.run(main())

#
# End of higgs_audio_demo.py
#######################################################################################################################