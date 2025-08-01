#!/usr/bin/env python3
"""
Test script for Logits Checker with OpenAI API
Tests the logprobs functionality end-to-end
"""

import json
import os
from typing import Dict, Any, List

# Add the project directory to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_chatbook.Chat.Chat_Functions import chat
from tldw_chatbook.config import load_settings

def test_openai_logprobs():
    """Test OpenAI API with logprobs enabled"""
    print("=== Testing OpenAI Logprobs Support ===\n")
    
    # Load settings to check if OpenAI is configured
    settings = load_settings()
    api_settings = settings.get('api_settings', {})
    openai_config = api_settings.get('openai', {})
    
    if not openai_config.get('api_key'):
        print("❌ OpenAI API key not found in settings")
        print("Please configure your OpenAI API key in ~/.config/tldw_cli/config.toml")
        return
    
    # Test parameters
    test_prompt = "The capital of France is"
    provider = "openai"
    model = "gpt-3.5-turbo"  # or gpt-4 if available
    
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Prompt: '{test_prompt}'")
    print(f"Logprobs: True")
    print(f"Top logprobs: 5")
    print("\n" + "="*50 + "\n")
    
    # Test streaming with logprobs
    print("Testing streaming response with logprobs...")
    try:
        result = chat(
            message=test_prompt,
            history=[],
            media_content=None,
            selected_parts=[],
            api_endpoint=provider,
            api_key=openai_config.get('api_key'),
            custom_prompt=None,
            temperature=0.7,
            streaming=True,
            model=model,
            max_tokens=10,
            llm_logprobs=True,
            llm_top_logprobs=5
        )
        
        print("\n✅ Streaming test completed")
        print(f"Response type: {type(result)}")
        
        # For streaming, result is a generator
        if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
            print("\nStreaming chunks:")
            chunk_count = 0
            for chunk in result:
                chunk_count += 1
                print(f"\nChunk {chunk_count}:")
                print(f"  Type: {type(chunk)}")
                if hasattr(chunk, 'text_chunk'):
                    print(f"  Text: '{chunk.text_chunk}'")
                if hasattr(chunk, 'logprobs'):
                    print(f"  Has logprobs: {chunk.logprobs is not None}")
                    if chunk.logprobs:
                        print(f"  Logprobs data: {json.dumps(chunk.logprobs, indent=2)[:200]}...")
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50 + "\n")
    
    # Test non-streaming with logprobs
    print("Testing non-streaming response with logprobs...")
    try:
        result = chat(
            message=test_prompt,
            history=[],
            media_content=None,
            selected_parts=[],
            api_endpoint=provider,
            api_key=openai_config.get('api_key'),
            custom_prompt=None,
            temperature=0.7,
            streaming=False,
            model=model,
            max_tokens=10,
            llm_logprobs=True,
            llm_top_logprobs=5
        )
        
        print("\n✅ Non-streaming test completed")
        print(f"Response: '{result}'")
        
    except Exception as e:
        print(f"❌ Non-streaming test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Summary ===")
    print("The tests above should show:")
    print("1. Streaming chunks with logprobs data attached")
    print("2. Each chunk containing token probabilities")
    print("3. Top 5 alternative tokens for each position")

if __name__ == "__main__":
    print("Starting OpenAI Logprobs Test...\n")
    test_openai_logprobs()