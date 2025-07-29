#!/usr/bin/env python3
"""
Test Logits Checker UI functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Test Instructions:")
print("1. Run the app with: python -m tldw_chatbook.app")
print("2. Navigate to Evals tab")
print("3. Click on 'Logits Checker'")
print("4. Select OpenAI as provider")
print("5. Select gpt-3.5-turbo as model")
print("6. Enter prompt: 'The capital of France is'")
print("7. Click 'Generate with Logits'")
print("8. You should see:")
print("   - Tokens appearing as clickable buttons")
print("   - Click a token to see its top alternatives with probabilities")
print("\nExpected output:")
print("- Token: 'Paris' with alternatives like ' Paris', ' ', etc.")
print("- Each alternative showing its probability percentage")
print("\nIf logprobs are working correctly, you'll see the alternatives table populate when clicking tokens.")