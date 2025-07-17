#!/usr/bin/env python3
"""
Test script for Moonshot AI integration.
This script tests the basic functionality of the Moonshot AI provider.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_moonshot
from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def test_moonshot_direct():
    """Test calling chat_with_moonshot directly."""
    print("\n=== Testing chat_with_moonshot directly ===")
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you tell me what 2+2 equals? Please respond in one sentence."}
    ]
    
    # You'll need to set your API key as an environment variable
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("ERROR: Please set MOONSHOT_API_KEY environment variable")
        return False
    
    try:
        # Test non-streaming
        print("\nTesting non-streaming mode...")
        response = chat_with_moonshot(
            input_data=messages,
            model="moonshot-v1-8k",
            api_key=api_key,
            streaming=False
        )
        
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            print("✓ Non-streaming test passed")
        else:
            print(f"✗ Non-streaming test failed: {response}")
            return False
            
        # Test streaming
        print("\nTesting streaming mode...")
        stream_response = chat_with_moonshot(
            input_data=messages,
            model="moonshot-v1-8k",
            api_key=api_key,
            streaming=True
        )
        
        print("Streaming response: ", end="", flush=True)
        for chunk in stream_response:
            if chunk.strip():
                # In real implementation, you'd parse the SSE data
                print(".", end="", flush=True)
        print("\n✓ Streaming test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during test: {type(e).__name__}: {e}")
        return False


def test_moonshot_via_chat_api_call():
    """Test calling Moonshot via the unified chat_api_call interface."""
    print("\n=== Testing Moonshot via chat_api_call ===")
    
    messages = [
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ]
    
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("ERROR: Please set MOONSHOT_API_KEY environment variable")
        return False
    
    try:
        response = chat_api_call(
            api_endpoint="moonshot",
            messages_payload=messages,
            api_key=api_key,
            model="moonshot-v1-8k",
            temp=0.7,
            streaming=False
        )
        
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            print("✓ chat_api_call test passed")
            return True
        else:
            print(f"✗ chat_api_call test failed: {response}")
            return False
            
    except Exception as e:
        print(f"✗ Error during test: {type(e).__name__}: {e}")
        return False


def test_moonshot_models():
    """Test different Moonshot models."""
    print("\n=== Testing different Moonshot models ===")
    
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("ERROR: Please set MOONSHOT_API_KEY environment variable")
        return False
    
    # Test a subset of models (testing all would be expensive)
    models = [
        "kimi-latest",           # Latest Kimi model
        "moonshot-v1-auto",      # Auto model selection
        "moonshot-v1-8k",        # Standard 8K model
        "moonshot-v1-32k",       # Standard 32K model
        # Optionally test vision models if needed
        # "moonshot-v1-8k-vision-preview",
    ]
    messages = [{"role": "user", "content": "Say 'Hello' in one word."}]
    
    all_passed = True
    for model in models:
        try:
            print(f"\nTesting model: {model}")
            response = chat_with_moonshot(
                input_data=messages,
                model=model,
                api_key=api_key,
                streaming=False,
                max_tokens=10
            )
            
            if response and "choices" in response:
                print(f"✓ {model} test passed")
            else:
                print(f"✗ {model} test failed")
                all_passed = False
                
        except Exception as e:
            print(f"✗ {model} test failed with error: {e}")
            all_passed = False
    
    return all_passed


def test_moonshot_vision():
    """Test Moonshot vision models (optional)."""
    print("\n=== Testing Moonshot vision capabilities ===")
    
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("ERROR: Please set MOONSHOT_API_KEY environment variable")
        return False
    
    # Test vision model with text-only input first
    messages = [{"role": "user", "content": "Describe what you see in detail."}]
    
    try:
        print("Testing moonshot-v1-8k-vision-preview with text...")
        response = chat_with_moonshot(
            input_data=messages,
            model="moonshot-v1-8k-vision-preview",
            api_key=api_key,
            streaming=False,
            max_tokens=50
        )
        
        if response and "choices" in response:
            print("✓ Vision model accepts text input")
            return True
        else:
            print("✗ Vision model test failed")
            return False
            
    except Exception as e:
        print(f"✗ Vision test failed with error: {e}")
        return False


def main():
    """Run all tests."""
    print("Moonshot AI Integration Test Suite")
    print("==================================")
    
    # Check for API key
    if not os.environ.get("MOONSHOT_API_KEY"):
        print("\nERROR: MOONSHOT_API_KEY environment variable not set!")
        print("Please set it with: export MOONSHOT_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Direct API Test", test_moonshot_direct),
        ("Chat API Call Test", test_moonshot_via_chat_api_call),
        ("Model Variants Test", test_moonshot_models),
        # ("Vision Model Test", test_moonshot_vision)  # Optional - uncomment to test
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Moonshot integration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()