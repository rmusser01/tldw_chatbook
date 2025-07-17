#!/usr/bin/env python3
"""
Test script for tool calling implementation.
Tests both the tool executor and the tool parsing functionality.
"""

import asyncio
import json
from tldw_chatbook.Tools import get_tool_executor
from tldw_chatbook.Chat.Chat_Functions import parse_tool_calls_from_response

async def test_tool_executor():
    """Test the tool executor with built-in tools"""
    print("=== Testing Tool Executor ===")
    
    executor = get_tool_executor()
    
    # Test 1: Calculator tool
    print("\n1. Testing Calculator Tool:")
    calc_call = {
        "id": "test_calc_1",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "42 * 3.14159"}'
        }
    }
    
    result = await executor.execute_tool_call(calc_call)
    print(f"   Result: {json.dumps(result, indent=2)}")
    
    # Test 2: DateTime tool
    print("\n2. Testing DateTime Tool:")
    datetime_call = {
        "id": "test_datetime_1",
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "arguments": '{}'
        }
    }
    
    result = await executor.execute_tool_call(datetime_call)
    print(f"   Result: {json.dumps(result, indent=2)}")
    
    # Test 3: Multiple tools concurrently
    print("\n3. Testing Multiple Tools Concurrently:")
    tool_calls = [
        {
            "id": "test_multi_1",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression": "100 / 25"}'
            }
        },
        {
            "id": "test_multi_2",
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "arguments": '{"timezone": "America/New_York"}'
            }
        }
    ]
    
    results = await executor.execute_tool_calls(tool_calls)
    print(f"   Results: {json.dumps(results, indent=2)}")
    
    # Test 4: Error handling
    print("\n4. Testing Error Handling:")
    error_call = {
        "id": "test_error_1",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "1/0"}'  # Division by zero
        }
    }
    
    result = await executor.execute_tool_call(error_call)
    print(f"   Result: {json.dumps(result, indent=2)}")

def test_tool_parsing():
    """Test tool call parsing from various response formats"""
    print("\n\n=== Testing Tool Call Parsing ===")
    
    # Test 1: OpenAI format
    print("\n1. Testing OpenAI Format:")
    openai_response = {
        "choices": [{
            "message": {
                "content": "I'll calculate that for you.",
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "2+2"}'
                    }
                }]
            }
        }]
    }
    
    tool_calls = parse_tool_calls_from_response(openai_response)
    print(f"   Parsed tool calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    # Test 2: Anthropic format
    print("\n2. Testing Anthropic Format:")
    anthropic_response = {
        "stop_reason": "tool_use",
        "content": [{
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "get_current_datetime",
            "input": {}
        }]
    }
    
    tool_calls = parse_tool_calls_from_response(anthropic_response)
    print(f"   Parsed tool calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    # Test 3: No tool calls
    print("\n3. Testing Response Without Tool Calls:")
    no_tools_response = {
        "choices": [{
            "message": {
                "content": "The answer is 42."
            }
        }]
    }
    
    tool_calls = parse_tool_calls_from_response(no_tools_response)
    print(f"   Parsed tool calls: {tool_calls}")

async def main():
    """Run all tests"""
    print("Tool Calling Implementation Test Suite\n")
    
    # Test tool executor
    await test_tool_executor()
    
    # Test tool parsing
    test_tool_parsing()
    
    print("\n\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(main())