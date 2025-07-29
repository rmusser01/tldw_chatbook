#!/usr/bin/env python3
"""
Test raw OpenAI streaming response with logprobs
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.config import load_settings

def test_raw_streaming():
    """Test raw streaming response"""
    print("=== Testing Raw OpenAI Streaming with Logprobs ===\n")
    
    # Load settings
    settings = load_settings()
    api_settings = settings.get('api_settings', {})
    openai_config = api_settings.get('openai', {})
    
    if not openai_config.get('api_key'):
        print("❌ OpenAI API key not found")
        return
    
    # Prepare test
    messages = [{"role": "user", "content": "The capital of France is"}]
    
    print("Making streaming API call with logprobs enabled...")
    response = chat_api_call(
        api_endpoint="openai",
        messages_payload=messages,
        api_key=openai_config.get('api_key'),
        temp=0.7,
        streaming=True,
        model="gpt-3.5-turbo",
        logprobs=True,
        top_logprobs=5,
        max_tokens=10
    )
    
    print(f"Response type: {type(response)}")
    print("\nParsing SSE chunks:\n")
    
    chunk_count = 0
    logprobs_found = False
    
    for chunk in response:
        chunk_count += 1
        
        # Skip empty chunks
        if not chunk or not chunk.strip():
            continue
            
        print(f"--- Chunk {chunk_count} ---")
        print(f"Raw: {chunk[:100]}...")
        
        # Parse SSE data
        if chunk.startswith("data: "):
            data_str = chunk[6:].strip()
            
            if data_str == "[DONE]":
                print("Stream complete")
                break
                
            try:
                data = json.loads(data_str)
                
                # Check for content
                if 'choices' in data and data['choices']:
                    choice = data['choices'][0]
                    
                    # Check delta content
                    if 'delta' in choice:
                        delta = choice['delta']
                        if 'content' in delta:
                            print(f"Content: '{delta['content']}'")
                    
                    # Check for logprobs
                    if 'logprobs' in choice and choice['logprobs']:
                        logprobs_found = True
                        print("✅ LOGPROBS FOUND!")
                        print(f"Logprobs: {json.dumps(choice['logprobs'], indent=2)}")
                    else:
                        print("❌ No logprobs in this chunk")
                        
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
        
        print()
        
        # Limit output
        if chunk_count > 10:
            print("... (limiting output)")
            break
    
    print(f"\n=== Summary ===")
    print(f"Total chunks: {chunk_count}")
    print(f"Logprobs found: {'✅ Yes' if logprobs_found else '❌ No'}")

if __name__ == "__main__":
    test_raw_streaming()