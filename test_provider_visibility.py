#!/usr/bin/env python3
"""
Quick diagnostic to check if lightning-whisper-mlx shows up in the providers list.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

print("Platform:", sys.platform)
print("Lightning Whisper MLX available:", DEPENDENCIES_AVAILABLE.get('lightning_whisper_mlx', False))

# Initialize service
service = TranscriptionService()

# Check providers
providers = service.get_available_providers()
print("\nAvailable providers:", providers)

# Check if it's in the list
if 'lightning-whisper-mlx' in providers:
    print("✅ lightning-whisper-mlx is in the providers list")
    
    # Check if it's the default
    print(f"\nDefault provider: {service.config['default_provider']}")
    
    # Get models
    models = service.list_available_models('lightning-whisper-mlx')
    print(f"\nModels for lightning-whisper-mlx: {len(models.get('lightning-whisper-mlx', []))} available")
else:
    print("❌ lightning-whisper-mlx is NOT in the providers list")
    
    # Debug why
    print("\nDebugging...")
    print("LIGHTNING_WHISPER_AVAILABLE:", 'LIGHTNING_WHISPER_AVAILABLE' in dir(service))