#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from tldw_chatbook.model_capabilities import get_model_capabilities

# Test the model capabilities
mc = get_model_capabilities()

# Check if the model is in direct mappings
print("Direct mapping for gpt-4.1-2025-04-14:", mc.direct_mappings.get('gpt-4.1-2025-04-14'))

# Check patterns for OpenAI
print("\nOpenAI patterns:")
if 'OpenAI' in mc._compiled_patterns:
    for pattern, caps in mc._compiled_patterns['OpenAI']:
        if pattern.match('gpt-4.1-2025-04-14'):
            print(f"  Pattern '{pattern.pattern}' matches: {caps}")

# Test is_vision_capable
result = mc.is_vision_capable('OpenAI', 'gpt-4.1-2025-04-14')
print(f"\nis_vision_capable('OpenAI', 'gpt-4.1-2025-04-14'): {result}")

# Clear cache and test again
mc.clear_cache()
result2 = mc.is_vision_capable('OpenAI', 'gpt-4.1-2025-04-14')
print(f"After cache clear: {result2}")