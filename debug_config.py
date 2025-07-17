#!/usr/bin/env python3
"""Debug config loading"""

import sys
sys.path.insert(0, '.')

from tldw_chatbook.config import load_cli_config_and_ensure_existence
from tldw_chatbook.model_capabilities import ModelCapabilities
import json

# Load the full config
print("Loading config...")
config = load_cli_config_and_ensure_existence()

# Check if model_capabilities section exists
if "model_capabilities" in config:
    print("\nFound model_capabilities section!")
    model_caps_config = config["model_capabilities"]
    
    # Check what's in it
    if "models" in model_caps_config:
        print(f"\nFound {len(model_caps_config['models'])} models:")
        for model_name in sorted(model_caps_config['models'].keys()):
            print(f"  - {model_name}: {model_caps_config['models'][model_name]}")
    
    if "patterns" in model_caps_config:
        print(f"\nFound patterns for {len(model_caps_config['patterns'])} providers")
        
else:
    print("\nNo model_capabilities section found in config!")
    print("\nAvailable top-level keys:")
    for key in sorted(config.keys()):
        print(f"  - {key}")

# Now test the ModelCapabilities class directly
print("\n" + "="*60)
print("Testing ModelCapabilities class...")

# Initialize with loaded config
if "model_capabilities" in config:
    caps = ModelCapabilities(config["model_capabilities"])
else:
    caps = ModelCapabilities()

# Test specific model
test_model = "gpt-4.1-2025-04-14"
provider = "OpenAI"

print(f"\nChecking {provider}/{test_model}:")
print(f"  Vision capable: {caps.is_vision_capable(provider, test_model)}")

full_caps = caps.get_model_capabilities(provider, test_model)
print(f"  Full capabilities: {json.dumps(full_caps, indent=4)}")

# Check what's in direct_mappings
print(f"\nDirect mappings has {len(caps.direct_mappings)} entries")
if test_model in caps.direct_mappings:
    print(f"  Found {test_model} in direct mappings: {caps.direct_mappings[test_model]}")
else:
    print(f"  {test_model} NOT found in direct mappings")