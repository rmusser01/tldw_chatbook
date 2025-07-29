#!/usr/bin/env python3
"""
Test script to check streaming logprobs parsing
"""

import json
from typing import Generator

# Add the project directory to Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Event_Handlers.worker_events import chat_wrapper_function
from tldw_chatbook.config import load_settings

def test_streaming_parser():
    """Test the streaming parser directly"""
    print("=== Testing Streaming Logprobs Parser ===\n")
    
    # Load settings
    settings = load_settings()
    api_settings = settings.get('api_settings', {})
    openai_config = api_settings.get('openai', {})
    
    if not openai_config.get('api_key'):
        print("❌ OpenAI API key not found")
        return
    
    # Prepare test input
    messages = [{"role": "user", "content": "The capital of France is"}]
    
    print("Testing direct API call with streaming...")
    raw_response = chat_api_call(
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
    
    print(f"Raw response type: {type(raw_response)}")
    
    if isinstance(raw_response, Generator):
        print("\nParsing streaming chunks through wrapper...")
        
        # Create wrapper generator
        wrapped_gen = chat_wrapper_function(raw_response, "openai")
        
        chunk_count = 0
        for event in wrapped_gen:
            chunk_count += 1
            print(f"\nChunk {chunk_count}:")
            print(f"  Type: {type(event)}")
            print(f"  Class: {event.__class__.__name__}")
            
            if hasattr(event, 'text_chunk'):
                print(f"  Text: '{event.text_chunk}'")
            
            if hasattr(event, 'logprobs'):
                print(f"  Has logprobs: {event.logprobs is not None}")
                if event.logprobs:
                    # Pretty print first 200 chars of logprobs
                    logprobs_str = json.dumps(event.logprobs, indent=2)
                    if len(logprobs_str) > 200:
                        logprobs_str = logprobs_str[:200] + "..."
                    print(f"  Logprobs:\n{logprobs_str}")
    
    print("\n=== Raw Stream Analysis ===")
    print("Testing raw SSE stream parsing...")
    
    # Get another raw stream
    raw_response2 = chat_api_call(
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
    
    # Look at raw chunks
    for i, chunk in enumerate(raw_response2):
        if i < 3:  # Just first 3 chunks
            print(f"\nRaw chunk {i+1}:")
            print(f"  Type: {type(chunk)}")
            print(f"  Content preview: {chunk[:100]}...")
            
            # Try to parse as SSE
            if chunk.startswith("data: "):
                data_str = chunk[6:].strip()
                if data_str and data_str != "[DONE]":
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and data['choices']:
                            choice = data['choices'][0]
                            if 'logprobs' in choice:
                                print(f"  ✅ Contains logprobs!")
                                print(f"  Logprobs: {json.dumps(choice['logprobs'], indent=2)[:100]}...")
                    except json.JSONDecodeError:
                        print(f"  ❌ Failed to parse JSON")

if __name__ == "__main__":
    test_streaming_parser()