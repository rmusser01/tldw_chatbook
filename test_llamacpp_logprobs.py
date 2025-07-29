#!/usr/bin/env python3
"""
Test script for Logits Checker with llama.cpp server
Tests the logprobs functionality with local llama.cpp server
"""

import json
import os
from typing import Dict, Any, List

# Add the project directory to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_chatbook.Chat.Chat_Functions import chat, chat_api_call
from tldw_chatbook.config import load_settings

def test_llamacpp_logprobs():
    """Test llama.cpp server with logprobs enabled"""
    print("=== Testing llama.cpp Logprobs Support ===\n")
    
    # Load settings to check if llama.cpp is configured
    settings = load_settings()
    api_settings = settings.get('api_settings', {})
    llama_config = api_settings.get('llama_cpp', {})
    
    # Set up llama.cpp connection
    api_url = "http://127.0.0.1:8080/"
    print(f"Server URL: {api_url}")
    print("Note: Make sure your llama.cpp server is running on port 8080\n")
    
    # Test parameters
    test_prompt = "The capital of France is"
    provider = "llama_cpp"
    
    print(f"Provider: {provider}")
    print(f"Prompt: '{test_prompt}'")
    print(f"Logprobs: True")
    print(f"Top logprobs: 5")
    print("\n" + "="*50 + "\n")
    
    # First, test direct API call to check raw response
    print("Testing direct API call with logprobs...")
    messages = [{"role": "user", "content": test_prompt}]
    
    try:
        # Test streaming with chat_api_call
        print("\n1. Testing streaming with chat_api_call...")
        response = chat_api_call(
            api_endpoint=provider,
            messages_payload=messages,
            api_key=None,  # llama.cpp doesn't need API key
            temp=0.7,
            streaming=True,
            model=None,  # llama.cpp serves one model at a time
            logprobs=True,
            top_logprobs=5,
            max_tokens=10
        )
        
        print(f"Response type: {type(response)}")
        
        if hasattr(response, '__iter__'):
            print("\nStreaming chunks:")
            chunk_count = 0
            logprobs_found = False
            
            for chunk in response:
                chunk_count += 1
                if chunk_count > 5:  # Limit output
                    print("... (limiting output)")
                    break
                    
                print(f"\nChunk {chunk_count}:")
                print(f"  Raw: {chunk[:100]}...")
                
                # Parse SSE data if it's a string
                if isinstance(chunk, str) and chunk.startswith("data: "):
                    data_str = chunk[6:].strip()
                    if data_str == "[DONE]":
                        print("  Stream complete")
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Check for logprobs
                        if 'choices' in data and data['choices']:
                            choice = data['choices'][0]
                            if 'logprobs' in choice and choice['logprobs']:
                                logprobs_found = True
                                print("  ✅ LOGPROBS FOUND!")
                                print(f"  Logprobs: {json.dumps(choice['logprobs'], indent=2)[:200]}...")
                            else:
                                print("  ❌ No logprobs in this chunk")
                                
                    except json.JSONDecodeError:
                        print("  Failed to parse JSON")
            
            print(f"\nLogprobs found in streaming: {'✅ Yes' if logprobs_found else '❌ No'}")
                
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50 + "\n")
    
    # Test non-streaming
    print("2. Testing non-streaming with chat_api_call...")
    try:
        response = chat_api_call(
            api_endpoint=provider,
            messages_payload=messages,
            api_key=None,
            temp=0.7,
            streaming=False,
            model=None,
            logprobs=True,
            top_logprobs=5,
            max_tokens=10
        )
        
        print(f"\n✅ Non-streaming test completed")
        print(f"Response type: {type(response)}")
        
        if isinstance(response, dict):
            # Check for logprobs in response
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if 'logprobs' in choice and choice['logprobs']:
                    print("✅ LOGPROBS FOUND in response!")
                    print(f"Logprobs: {json.dumps(choice['logprobs'], indent=2)[:500]}...")
                else:
                    print("❌ No logprobs in response")
                    print(f"Choice keys: {choice.keys()}")
            
            # Show the response text
            if 'choices' in response and response['choices']:
                content = response['choices'][0].get('message', {}).get('content', '')
                print(f"\nGenerated text: '{content}'")
        
    except Exception as e:
        print(f"❌ Non-streaming test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50 + "\n")
    
    # Test with the high-level chat function
    print("3. Testing with high-level chat function...")
    try:
        result = chat(
            message=test_prompt,
            history=[],
            media_content=None,
            selected_parts=[],
            api_endpoint=provider,
            api_key=None,
            custom_prompt=None,
            temperature=0.7,
            streaming=False,
            model=None,
            max_tokens=10,
            llm_logprobs=True,
            llm_top_logprobs=5
        )
        
        print("\n✅ Chat function test completed")
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            if 'choices' in result and result['choices']:
                choice = result['choices'][0]
                if 'logprobs' in choice:
                    print("✅ Logprobs present in chat response")
                else:
                    print("❌ No logprobs in chat response")
        
    except Exception as e:
        print(f"❌ Chat function test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Summary ===")
    print("The tests above should show:")
    print("1. Whether llama.cpp server returns logprobs in streaming mode")
    print("2. Whether llama.cpp server returns logprobs in non-streaming mode")
    print("3. Integration with the high-level chat function")
    print("\nNOTE: Your llama.cpp server must be compiled with logprobs support")
    print("and running with appropriate flags (e.g., --logits-all)")

if __name__ == "__main__":
    print("Starting llama.cpp Logprobs Test...\n")
    print("Prerequisites:")
    print("1. llama.cpp server running on http://127.0.0.1:8080")
    print("2. Server started with logprobs support (--logits-all flag)")
    print("3. A model loaded in the server")
    print("\n")
    
    test_llamacpp_logprobs()