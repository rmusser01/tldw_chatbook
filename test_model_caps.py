#!/usr/bin/env python3
"""Test model capabilities loading"""

import sys
sys.path.insert(0, '.')

from tldw_chatbook.model_capabilities import get_model_capabilities, is_vision_capable

# Test the models
test_models = [
    ("OpenAI", "gpt-4.1-2025-04-14"),
    ("OpenAI", "o4-mini-2025-04-16"),
    ("OpenAI", "o3-2025-04-16"),
    ("OpenAI", "o3-mini-2025-01-31"),
    ("OpenAI", "gpt-4.1-mini-2025-04-14"),
    ("OpenAI", "gpt-4.1-nano-2025-04-14"),
    ("OpenAI", "gpt-4o"),
    ("OpenAI", "gpt-3.5-turbo"),
]

print("Testing model capabilities...")
print("-" * 60)

for provider, model in test_models:
    vision_capable = is_vision_capable(provider, model)
    print(f"{provider}/{model}: vision={vision_capable}")
    
print("-" * 60)

# Get full capabilities for one model
caps = get_model_capabilities()
full_caps = caps.get_model_capabilities("OpenAI", "gpt-4.1-2025-04-14")
print(f"\nFull capabilities for OpenAI/gpt-4.1-2025-04-14:")
for key, value in full_caps.items():
    print(f"  {key}: {value}")