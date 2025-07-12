# tool_message_widgets.py
"""
Specialized message widgets for displaying tool calls and tool results in the chat interface.
"""

import json
from typing import List, Dict, Any, Optional

from rich.text import Text
from textual.widgets import Static, Label
from textual.containers import Vertical, Horizontal
from textual.css.query import QueryError

from .chat_message import ChatMessage


class ToolCallMessage(ChatMessage):
    """Widget for displaying tool/function calls in the chat."""
    
    def __init__(
        self,
        tool_calls: List[Dict[str, Any]],
        message_id: Optional[str] = None,
        generation_complete: bool = True,
        **kwargs
    ):
        """
        Initialize a tool call message widget.
        
        Args:
            tool_calls: List of tool calls in OpenAI format
            message_id: Optional message ID for database tracking
            generation_complete: Whether the message is complete
            **kwargs: Additional arguments passed to ChatMessage
        """
        self.tool_calls = tool_calls
        
        # Format the tool calls for display
        formatted_content = self._format_tool_calls()
        
        # Initialize parent with formatted content
        super().__init__(
            message=formatted_content,
            role="Tool Call",
            generation_complete=generation_complete,
            message_id=message_id,
            **kwargs
        )
        
        # Add special CSS class for styling
        self.add_class("-tool-call")
    
    def _format_tool_calls(self) -> str:
        """Format tool calls for display."""
        if not self.tool_calls:
            return "[No tool calls]"
        
        lines = []
        lines.append("🔧 [bold cyan]Tool Calls:[/bold cyan]")
        
        for i, call in enumerate(self.tool_calls, 1):
            func = call.get("function", {})
            func_name = func.get("name", "unknown")
            
            lines.append(f"\n[yellow]#{i} {func_name}[/yellow]")
            
            # Parse and format arguments
            try:
                args_str = func.get("arguments", "{}")
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                
                if args:
                    lines.append("[dim]Arguments:[/dim]")
                    for key, value in args.items():
                        # Truncate long values
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."
                        lines.append(f"  [dim]• {key}: {value_str}[/dim]")
                else:
                    lines.append("  [dim](no arguments)[/dim]")
            except json.JSONDecodeError:
                lines.append(f"  [dim red]Invalid arguments: {func.get('arguments', 'N/A')}[/dim red]")
        
        return "\n".join(lines)


class ToolResultMessage(ChatMessage):
    """Widget for displaying tool execution results in the chat."""
    
    def __init__(
        self,
        tool_results: List[Dict[str, Any]],
        message_id: Optional[str] = None,
        generation_complete: bool = True,
        **kwargs
    ):
        """
        Initialize a tool result message widget.
        
        Args:
            tool_results: List of tool execution results
            message_id: Optional message ID for database tracking
            generation_complete: Whether the message is complete
            **kwargs: Additional arguments passed to ChatMessage
        """
        self.tool_results = tool_results
        
        # Format the results for display
        formatted_content = self._format_results()
        
        # Initialize parent with formatted content
        super().__init__(
            message=formatted_content,
            role="Tool Result",
            generation_complete=generation_complete,
            message_id=message_id,
            **kwargs
        )
        
        # Add special CSS class for styling
        self.add_class("-tool-result")
    
    def _format_results(self) -> str:
        """Format tool results for display."""
        if not self.tool_results:
            return "[No results]"
        
        lines = []
        lines.append("📊 [bold green]Tool Results:[/bold green]")
        
        for i, result in enumerate(self.tool_results, 1):
            tool_call_id = result.get("tool_call_id", f"call_{i}")
            
            if "error" in result:
                # Error result
                lines.append(f"\n[red]#{i} Error (ID: {tool_call_id}):[/red]")
                lines.append(f"  [dim red]{result['error']}[/dim red]")
            else:
                # Success result
                lines.append(f"\n[green]#{i} Success (ID: {tool_call_id}):[/green]")
                
                result_data = result.get("result", {})
                if isinstance(result_data, dict):
                    # Format dict results
                    for key, value in result_data.items():
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."
                        lines.append(f"  [dim]• {key}: {value_str}[/dim]")
                elif isinstance(result_data, (list, tuple)):
                    # Format list results
                    lines.append(f"  [dim]{len(result_data)} items returned[/dim]")
                    for item in result_data[:3]:  # Show first 3 items
                        item_str = str(item)
                        if len(item_str) > 80:
                            item_str = item_str[:77] + "..."
                        lines.append(f"    [dim]- {item_str}[/dim]")
                    if len(result_data) > 3:
                        lines.append(f"    [dim]... and {len(result_data) - 3} more[/dim]")
                else:
                    # Simple result
                    result_str = str(result_data)
                    if len(result_str) > 200:
                        result_str = result_str[:197] + "..."
                    lines.append(f"  [dim]{result_str}[/dim]")
        
        return "\n".join(lines)


class ToolExecutionWidget(Vertical):
    """
    Container widget that displays both tool calls and their results together.
    """
    
    def __init__(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize a tool execution widget.
        
        Args:
            tool_calls: List of tool calls
            tool_results: Optional list of tool results (can be added later)
            **kwargs: Additional arguments passed to Vertical
        """
        super().__init__(**kwargs)
        
        self.tool_calls = tool_calls
        self.tool_results = tool_results or []
        
        # Create and mount the tool call message
        self.tool_call_widget = ToolCallMessage(tool_calls)
        
        # Create result widget if results are provided
        self.tool_result_widget = None
        if self.tool_results:
            self.tool_result_widget = ToolResultMessage(self.tool_results)
    
    def compose(self):
        """Compose the widget with tool call and optional result messages."""
        yield self.tool_call_widget
        if self.tool_result_widget:
            yield self.tool_result_widget
    
    def update_results(self, tool_results: List[Dict[str, Any]]):
        """
        Update the widget with tool execution results.
        
        Args:
            tool_results: List of tool execution results
        """
        self.tool_results = tool_results
        
        if not self.tool_result_widget:
            # Create and mount new result widget
            self.tool_result_widget = ToolResultMessage(tool_results)
            self.mount(self.tool_result_widget)
        else:
            # Update existing widget
            self.tool_result_widget.tool_results = tool_results
            self.tool_result_widget.message = self.tool_result_widget._format_results()
            # Update the display
            try:
                static_widget = self.tool_result_widget.query_one(".message-text", Static)
                static_widget.update(Text.from_markup(self.tool_result_widget.message))
            except QueryError:
                pass