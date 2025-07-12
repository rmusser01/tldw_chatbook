"""Tests for tool message widgets (ToolCallMessage, ToolResultMessage, ToolExecutionWidget)."""

import json
import pytest
from typing import List, Dict, Any

from textual.app import App
from textual.widgets import Static
from textual.css.query import QueryError

from tldw_chatbook.Widgets.tool_message_widgets import (
    ToolCallMessage,
    ToolResultMessage,
    ToolExecutionWidget
)


class ToolMessageTestApp(App):
    """Test app for mounting and testing tool message widgets."""
    
    def __init__(self, widget):
        super().__init__()
        self.test_widget = widget
    
    def compose(self):
        yield self.test_widget


@pytest.fixture
def empty_tool_calls():
    """Fixture for empty tool calls."""
    return []


@pytest.fixture
def single_tool_call():
    """Fixture for a single tool call."""
    return [{
        "function": {
            "name": "calculate",
            "arguments": json.dumps({"x": 5, "y": 10})
        }
    }]


@pytest.fixture
def multiple_tool_calls():
    """Fixture for multiple tool calls."""
    return [
        {
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "New York"})
            }
        },
        {
            "function": {
                "name": "get_time",
                "arguments": json.dumps({"timezone": "EST"})
            }
        }
    ]


@pytest.fixture
def success_tool_result():
    """Fixture for a successful tool result."""
    return [{
        "tool_call_id": "call_123",
        "result": {"answer": 42, "status": "success"}
    }]


@pytest.fixture
def error_tool_result():
    """Fixture for an error tool result."""
    return [{
        "tool_call_id": "call_456",
        "error": "Division by zero"
    }]


class TestToolCallMessage:
    """Tests for ToolCallMessage widget."""
    
    def test_init_with_empty_tool_calls(self, empty_tool_calls):
        """Test initialization with empty tool calls list."""
        widget = ToolCallMessage(tool_calls=empty_tool_calls, message_id="test_id")
        assert widget.tool_calls == []
        assert "[No tool calls]" in widget.message_text
        assert "Tool Call" in widget.role
        assert widget.message_id_internal == "test_id"
        assert "-tool-call" in widget.classes
    
    def test_init_with_single_tool_call(self, single_tool_call):
        """Test initialization with a single tool call."""
        widget = ToolCallMessage(tool_calls=single_tool_call)
        
        # Check formatting
        assert "Tool Calls:" in widget.message_text
        assert "calculate" in widget.message_text
        assert "x: 5" in widget.message_text
        assert "y: 10" in widget.message_text
    
    def test_init_with_multiple_tool_calls(self, multiple_tool_calls):
        """Test initialization with multiple tool calls."""
        widget = ToolCallMessage(tool_calls=multiple_tool_calls)
        
        # Check both functions appear
        assert "#1 get_weather" in widget.message_text
        assert "#2 get_time" in widget.message_text
        assert "location: New York" in widget.message_text
        assert "timezone: EST" in widget.message_text
    
    def test_format_tool_calls_with_no_arguments(self):
        """Test formatting when tool call has no arguments."""
        tool_calls = [{
            "function": {
                "name": "get_current_time",
                "arguments": "{}"
            }
        }]
        widget = ToolCallMessage(tool_calls=tool_calls)
        assert "(no arguments)" in widget.message_text
    
    def test_format_tool_calls_with_invalid_json(self):
        """Test formatting when tool call has invalid JSON arguments."""
        tool_calls = [{
            "function": {
                "name": "broken_function",
                "arguments": "invalid json {{"
            }
        }]
        widget = ToolCallMessage(tool_calls=tool_calls)
        assert "Invalid arguments:" in widget.message_text
        assert "invalid json {{" in widget.message_text
    
    def test_format_tool_calls_with_long_values(self):
        """Test truncation of long argument values."""
        long_text = "x" * 200
        tool_calls = [{
            "function": {
                "name": "process_text",
                "arguments": json.dumps({"text": long_text})
            }
        }]
        widget = ToolCallMessage(tool_calls=tool_calls)
        
        # Should be truncated to 100 chars (97 + "...")
        assert "..." in widget.message_text
        text_line = [line for line in widget.message_text.split("\n") if "text:" in line][0]
        # Extract just the value part without the [/dim] tag
        text_value = text_line.split("text: ")[1].replace("[/dim]", "")
        assert len(text_value) <= 100
    
    def test_generation_complete_flag(self):
        """Test generation_complete flag is properly set."""
        widget = ToolCallMessage(tool_calls=[], generation_complete=False)
        assert widget.generation_complete is False
        
        widget2 = ToolCallMessage(tool_calls=[], generation_complete=True)
        assert widget2.generation_complete is True
    
    @pytest.mark.asyncio
    async def test_widget_mounting(self, single_tool_call):
        """Test that widget can be mounted in an app."""
        widget = ToolCallMessage(tool_calls=single_tool_call)
        app = ToolMessageTestApp(widget)
        
        async with app.run_test() as pilot:
            # Verify widget is mounted
            assert widget.is_mounted
            # Verify widget has expected content
            assert "calculate" in widget.message_text


class TestToolResultMessage:
    """Tests for ToolResultMessage widget."""
    
    def test_init_with_empty_results(self):
        """Test initialization with empty results list."""
        widget = ToolResultMessage(tool_results=[], message_id="result_id")
        assert widget.tool_results == []
        assert "[No results]" in widget.message_text
        assert "Tool Result" in widget.role
        assert widget.message_id_internal == "result_id"
        assert "-tool-result" in widget.classes
    
    def test_init_with_success_result(self, success_tool_result):
        """Test initialization with a successful result."""
        widget = ToolResultMessage(tool_results=success_tool_result)
        
        assert "Tool Results:" in widget.message_text
        assert "Success" in widget.message_text
        assert "call_123" in widget.message_text
        assert "answer: 42" in widget.message_text
        assert "status: success" in widget.message_text
    
    def test_init_with_error_result(self, error_tool_result):
        """Test initialization with an error result."""
        widget = ToolResultMessage(tool_results=error_tool_result)
        
        assert "Error" in widget.message_text
        assert "call_456" in widget.message_text
        assert "Division by zero" in widget.message_text
    
    def test_format_results_with_list_data(self):
        """Test formatting when result contains a list."""
        tool_results = [{
            "tool_call_id": "call_789",
            "result": ["item1", "item2", "item3", "item4", "item5"]
        }]
        widget = ToolResultMessage(tool_results=tool_results)
        
        assert "5 items returned" in widget.message_text
        assert "item1" in widget.message_text
        assert "item2" in widget.message_text
        assert "item3" in widget.message_text
        assert "... and 2 more" in widget.message_text
    
    def test_multiple_results_mixed_types(self):
        """Test formatting multiple results with mixed success/error."""
        tool_results = [
            {
                "tool_call_id": "call_1",
                "result": {"status": "ok"}
            },
            {
                "tool_call_id": "call_2",
                "error": "Network timeout"
            },
            {
                "tool_call_id": "call_3",
                "result": [1, 2, 3]
            }
        ]
        widget = ToolResultMessage(tool_results=tool_results)
        
        assert "#1 Success" in widget.message_text
        assert "#2 Error" in widget.message_text
        assert "#3 Success" in widget.message_text
        assert "Network timeout" in widget.message_text
        assert "3 items returned" in widget.message_text
    
    @pytest.mark.asyncio
    async def test_widget_mounting(self, success_tool_result):
        """Test that result widget can be mounted in an app."""
        widget = ToolResultMessage(tool_results=success_tool_result)
        app = ToolMessageTestApp(widget)
        
        async with app.run_test() as pilot:
            # Verify widget is mounted
            assert widget.is_mounted
            # Verify widget has expected content
            assert "answer: 42" in widget.message_text


class TestToolExecutionWidget:
    """Tests for ToolExecutionWidget container."""
    
    def test_init_with_calls_only(self, single_tool_call):
        """Test initialization with only tool calls."""
        widget = ToolExecutionWidget(tool_calls=single_tool_call)
        
        assert widget.tool_calls == single_tool_call
        assert widget.tool_results == []
        assert widget.tool_call_widget is not None
        assert widget.tool_result_widget is None
    
    def test_init_with_calls_and_results(self, single_tool_call, success_tool_result):
        """Test initialization with both calls and results."""
        widget = ToolExecutionWidget(
            tool_calls=single_tool_call,
            tool_results=success_tool_result
        )
        
        assert widget.tool_calls == single_tool_call
        assert widget.tool_results == success_tool_result
        assert widget.tool_call_widget is not None
        assert widget.tool_result_widget is not None
    
    @pytest.mark.asyncio
    async def test_compose_with_calls_only(self, single_tool_call):
        """Test compose yields only call widget when no results."""
        widget = ToolExecutionWidget(tool_calls=single_tool_call)
        
        app = ToolMessageTestApp(widget)
        async with app.run_test() as pilot:
            # Should have one ToolCallMessage child
            children = list(widget.children)
            assert len(children) == 1
            assert isinstance(children[0], ToolCallMessage)
    
    @pytest.mark.asyncio
    async def test_update_results_creates_widget(self, single_tool_call):
        """Test update_results creates result widget if not exists."""
        widget = ToolExecutionWidget(tool_calls=single_tool_call)
        
        app = ToolMessageTestApp(widget)
        async with app.run_test() as pilot:
            # Initially only call widget
            assert len(list(widget.children)) == 1
            
            # Update with results
            tool_results = [{
                "tool_call_id": "async_1",
                "result": "completed"
            }]
            widget.update_results(tool_results)
            
            # Should now have both widgets
            await pilot.pause()
            assert len(list(widget.children)) == 2
            assert widget.tool_result_widget is not None
            assert isinstance(widget.tool_result_widget, ToolResultMessage)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_tool_call_with_nested_complex_data(self):
        """Test handling deeply nested complex data structures."""
        complex_args = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": ["a", "b", "c"],
                        "numbers": [1, 2, 3],
                        "mixed": [{"x": 1}, {"y": 2}]
                    }
                }
            }
        }
        tool_calls = [{
            "function": {
                "name": "complex_function",
                "arguments": json.dumps(complex_args)
            }
        }]
        
        widget = ToolCallMessage(tool_calls=tool_calls)
        # Should format without error
        assert "level1:" in widget.message_text
    
    def test_unicode_handling(self):
        """Test handling of unicode characters in tool calls and results."""
        tool_calls = [{
            "function": {
                "name": "translate",
                "arguments": json.dumps({
                    "text": "Hello 世界 🌍",
                    "emoji": "😀🎉🔧"
                })
            }
        }]
        tool_results = [{
            "tool_call_id": "unicode_1",
            "result": "Привет мир 🌏"
        }]
        
        call_widget = ToolCallMessage(tool_calls=tool_calls)
        result_widget = ToolResultMessage(tool_results=tool_results)
        
        # Should handle unicode without error
        assert "世界" in call_widget.message_text
        assert "😀" in call_widget.message_text
        assert "Привет" in result_widget.message_text
        assert "🌏" in result_widget.message_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
