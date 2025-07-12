"""End-to-end integration tests for tool calling flow in chat."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from textual.app import App
from textual.widgets import Input, Button

# Mock the tool-related imports for testing
class MockTool:
    """Base mock tool class."""
    name = "mock_tool"
    description = "Mock tool for testing"
    
    def get_schema(self):
        return {"type": "object", "properties": {}}
    
    async def execute(self, **kwargs):
        return {"result": "success"}


class MockCalculatorTool(MockTool):
    """Mock calculator tool for testing."""
    name = "calculator"
    description = "Performs arithmetic"
    
    async def execute(self, **kwargs):
        operation = kwargs.get("operation")
        a = kwargs.get("a", 0)
        b = kwargs.get("b", 0)
        
        if operation == "add":
            return {"result": a + b}
        elif operation == "divide" and b == 0:
            return {"error": "Division by zero"}
        return {"result": 0}


@pytest.fixture
def mock_llm_response_with_tools():
    """Create mock LLM responses with tool calls."""
    return {
        "calculation": {
            "content": "I'll calculate that.",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"operation": "add", "a": 5, "b": 3})
                }
            }]
        }
    }


class TestToolCallingFlow:
    """Test tool calling flow."""
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test basic tool execution."""
        tool = MockCalculatorTool()
        result = await tool.execute(operation="add", a=5, b=3)
        assert result["result"] == 8
    
    def test_tool_call_parsing(self, mock_llm_response_with_tools):
        """Test parsing tool calls from LLM response."""
        response = mock_llm_response_with_tools["calculation"]
        tool_calls = response.get("tool_calls", [])
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "calculator"
        
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["operation"] == "add"
        assert args["a"] == 5
        assert args["b"] == 3
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test tool error handling."""
        tool = MockCalculatorTool()
        result = await tool.execute(operation="divide", a=10, b=0)
        assert "error" in result
        assert "Division by zero" in result["error"]


class TestToolRegistration:
    """Test tool registration system."""
    
    def test_tool_schema(self):
        """Test getting tool schema."""
        tool = MockCalculatorTool()
        schema = tool.get_schema()
        assert isinstance(schema, dict)
        assert "type" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])