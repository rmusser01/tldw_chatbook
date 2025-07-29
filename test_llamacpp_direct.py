#!/usr/bin/env python3
"""
Direct test of llama.cpp server API
"""

import requests
import json

def test_direct():
    """Test llama.cpp server directly"""
    
    # Test 1: Check server is running
    print("=== Testing llama.cpp Server Directly ===\n")
    
    base_url = "http://127.0.0.1:8080"
    
    # Test server health
    print(f"1. Testing server at {base_url}")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Health check: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Server is running")
    except Exception as e:
        print(f"   ❌ Server not responding: {e}")
        return
    
    # Test models endpoint
    print("\n2. Checking models endpoint")
    try:
        response = requests.get(f"{base_url}/v1/models")
        print(f"   Models endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test chat completions with logprobs
    print("\n3. Testing chat completions with logprobs")
    
    payload = {
        "messages": [
            {"role": "user", "content": "The capital of France is"}
        ],
        "temperature": 0.7,
        "max_tokens": 10,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Request successful")
            
            # Check for logprobs
            if 'choices' in data and data['choices']:
                choice = data['choices'][0]
                if 'logprobs' in choice:
                    print("   ✅ LOGPROBS FOUND!")
                    print(f"   Logprobs: {json.dumps(choice['logprobs'], indent=2)[:300]}...")
                else:
                    print("   ❌ No logprobs in response")
                    print(f"   Choice keys: {choice.keys()}")
                
                # Show generated text
                if 'message' in choice:
                    print(f"   Generated: '{choice['message']['content']}'")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test streaming
    print("\n4. Testing streaming with logprobs")
    payload['stream'] = True
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            stream=True
        )
        
        if response.status_code == 200:
            print("   ✅ Streaming request successful")
            chunk_count = 0
            logprobs_found = False
            
            for line in response.iter_lines():
                if line:
                    chunk_count += 1
                    if chunk_count > 3:  # Just show first few
                        break
                    
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            print(f"\n   Chunk {chunk_count}:")
                            
                            if 'choices' in data and data['choices']:
                                choice = data['choices'][0]
                                if 'logprobs' in choice:
                                    logprobs_found = True
                                    print("   ✅ Has logprobs")
                                else:
                                    print("   ❌ No logprobs")
                                    
                        except json.JSONDecodeError:
                            print(f"   Failed to parse: {line[:50]}...")
            
            print(f"\n   Logprobs in streaming: {'✅ Yes' if logprobs_found else '❌ No'}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")

if __name__ == "__main__":
    test_direct()