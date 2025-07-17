#!/usr/bin/env python3
"""Simple import test"""

try:
    print("Testing imports...")
    from tldw_chatbook.Tools import get_tool_executor
    print("✓ Successfully imported get_tool_executor")
    
    from tldw_chatbook.Chat.Chat_Functions import parse_tool_calls_from_response
    print("✓ Successfully imported parse_tool_calls_from_response")
    
    from tldw_chatbook.Widgets.tool_message_widgets import ToolExecutionWidget
    print("✓ Successfully imported ToolExecutionWidget")
    
    # Try to get the tool executor
    executor = get_tool_executor()
    print(f"✓ Successfully created tool executor with {len(executor.tools)} tools")
    print(f"  Available tools: {list(executor.tools.keys())}")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()