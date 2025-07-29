#!/usr/bin/env python3
"""
Check llama.cpp configuration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_chatbook.config import load_settings

settings = load_settings()
api_settings = settings.get('api_settings', {})
llama_config = api_settings.get('llama_cpp', {})

print("=== llama.cpp Configuration ===")
print(f"Config keys: {llama_config.keys()}")
print(f"API URL: {llama_config.get('api_url', 'Not set')}")
print(f"Temperature: {llama_config.get('temperature', 'Not set')}")
print(f"Model: {llama_config.get('model', 'Not set')}")

# Also check if there's a local_llamacpp config
local_llama_config = api_settings.get('local_llamacpp', {})
if local_llama_config:
    print("\n=== local_llamacpp Configuration ===")
    print(f"Config keys: {local_llama_config.keys()}")
    print(f"API URL: {local_llama_config.get('api_url', 'Not set')}")