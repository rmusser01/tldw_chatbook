# tool_executor.py
"""
Tool execution framework for handling function/tool calls from LLMs.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from loguru import logger


class Tool(ABC):
    """Base class for all tools that can be called by LLMs."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool as it will be called by the LLM."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for the tool's parameters."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Dictionary with the result or error
        """
        pass
    
    def to_openai_format(self) -> dict:
        """Convert tool to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolExecutor:
    """Manages tool registration and execution with safety measures."""
    
    def __init__(self, timeout_seconds: int = 30, max_workers: int = 4):
        """
        Initialize the tool executor.
        
        Args:
            timeout_seconds: Maximum execution time for each tool
            max_workers: Maximum concurrent tool executions
        """
        self.tools: Dict[str, Tool] = {}
        self.timeout_seconds = timeout_seconds
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._execution_history: List[Dict[str, Any]] = []
    
    def register_tool(self, tool: Tool):
        """
        Register a tool for execution.
        
        Args:
            tool: The tool instance to register
        """
        if not isinstance(tool, Tool):
            raise ValueError(f"Tool must be an instance of Tool class, got {type(tool)}")
        
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str):
        """Remove a tool from the registry."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
    
    def get_available_tools(self) -> List[dict]:
        """Get list of available tools in OpenAI format."""
        return [tool.to_openai_format() for tool in self.tools.values()]
    
    async def execute_tool_call(self, tool_call: dict) -> dict:
        """
        Execute a single tool call with safety measures.
        
        Args:
            tool_call: Tool call in OpenAI format
            
        Returns:
            Dictionary with tool_call_id and either result or error
        """
        tool_call_id = tool_call.get("id", f"call_{datetime.now().timestamp()}")
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name")
        
        # Record execution start
        execution_record = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "start_time": datetime.now().isoformat(),
            "status": "started"
        }
        self._execution_history.append(execution_record)
        
        # Validate tool exists
        if not tool_name or tool_name not in self.tools:
            error_msg = f"Unknown tool: {tool_name}"
            logger.error(error_msg)
            execution_record["status"] = "error"
            execution_record["error"] = error_msg
            execution_record["end_time"] = datetime.now().isoformat()
            return {
                "tool_call_id": tool_call_id,
                "error": error_msg
            }
        
        # Parse arguments
        try:
            args_str = function_data.get("arguments", "{}")
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON arguments: {e}"
            logger.error(f"Tool {tool_name} - {error_msg}")
            execution_record["status"] = "error"
            execution_record["error"] = error_msg
            execution_record["end_time"] = datetime.now().isoformat()
            return {
                "tool_call_id": tool_call_id,
                "error": error_msg
            }
        
        # Execute with timeout
        tool = self.tools[tool_name]
        try:
            logger.info(f"Executing tool {tool_name} with args: {args}")
            
            # Run the async tool in the executor with timeout
            result = await asyncio.wait_for(
                tool.execute(**args),
                timeout=self.timeout_seconds
            )
            
            execution_record["status"] = "success"
            execution_record["end_time"] = datetime.now().isoformat()
            execution_record["result"] = result
            
            logger.info(f"Tool {tool_name} completed successfully")
            return {
                "tool_call_id": tool_call_id,
                "result": result
            }
            
        except asyncio.TimeoutError:
            error_msg = f"Tool execution timed out after {self.timeout_seconds} seconds"
            logger.error(f"Tool {tool_name} - {error_msg}")
            execution_record["status"] = "timeout"
            execution_record["error"] = error_msg
            execution_record["end_time"] = datetime.now().isoformat()
            return {
                "tool_call_id": tool_call_id,
                "error": error_msg
            }
        
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"Tool {tool_name} - {error_msg}", exc_info=True)
            execution_record["status"] = "error"
            execution_record["error"] = error_msg
            execution_record["end_time"] = datetime.now().isoformat()
            return {
                "tool_call_id": tool_call_id,
                "error": error_msg
            }
    
    async def execute_tool_calls(self, tool_calls: List[dict]) -> List[dict]:
        """
        Execute multiple tool calls concurrently.
        
        Args:
            tool_calls: List of tool calls in OpenAI format
            
        Returns:
            List of results with tool_call_id and either result or error
        """
        if not tool_calls:
            return []
        
        # Execute all tools concurrently
        tasks = [self.execute_tool_call(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred during gathering
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_call_id = tool_calls[i].get("id", f"call_{i}")
                processed_results.append({
                    "tool_call_id": tool_call_id,
                    "error": f"Execution failed: {str(result)}"
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent tool execution history."""
        return self._execution_history[-limit:]
    
    def clear_history(self):
        """Clear execution history."""
        self._execution_history.clear()
    
    def __del__(self):
        """Cleanup executor on deletion."""
        self.executor.shutdown(wait=False)


# Built-in tools

class DateTimeTool(Tool):
    """Tool for getting current date and time."""
    
    @property
    def name(self) -> str:
        return "get_current_datetime"
    
    @property
    def description(self) -> str:
        return "Get the current date and time in a specific timezone"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g., 'UTC', 'America/New_York'). Defaults to UTC.",
                    "default": "UTC"
                }
            },
            "required": []
        }
    
    async def execute(self, timezone: str = "UTC") -> Dict[str, Any]:
        """Get current datetime in specified timezone."""
        from zoneinfo import ZoneInfo
        
        try:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return {
                "datetime": now.isoformat(),
                "timezone": timezone,
                "date": now.date().isoformat(),
                "time": now.time().isoformat(),
                "weekday": now.strftime("%A"),
                "unix_timestamp": int(now.timestamp())
            }
        except Exception as e:
            raise ValueError(f"Invalid timezone '{timezone}': {e}")


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Perform mathematical calculations"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, expression: str) -> Dict[str, Any]:
        """Evaluate a mathematical expression safely."""
        import ast
        import operator
        
        # Allowed operators for safety
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.Mod: operator.mod,
        }
        
        # Allowed functions
        allowed_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
        }
        
        def safe_eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                op = type(node.op)
                if op not in allowed_operators:
                    raise ValueError(f"Operator {op.__name__} not allowed")
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                return allowed_operators[op](left, right)
            elif isinstance(node, ast.UnaryOp):
                op = type(node.op)
                if op not in allowed_operators:
                    raise ValueError(f"Operator {op.__name__} not allowed")
                operand = safe_eval(node.operand)
                return allowed_operators[op](operand)
            elif isinstance(node, ast.Call):
                func_name = node.func.id if isinstance(node.func, ast.Name) else None
                if func_name not in allowed_functions:
                    raise ValueError(f"Function {func_name} not allowed")
                args = [safe_eval(arg) for arg in node.args]
                return allowed_functions[func_name](*args)
            else:
                raise ValueError(f"Expression type {type(node).__name__} not allowed")
        
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            result = safe_eval(tree.body)
            
            return {
                "expression": expression,
                "result": result,
                "result_type": type(result).__name__
            }
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")


# Create a global tool executor instance
_global_executor = None

def get_tool_executor() -> ToolExecutor:
    """Get the global tool executor instance."""
    global _global_executor
    if _global_executor is None:
        _global_executor = ToolExecutor()
        # Register built-in tools
        _global_executor.register_tool(DateTimeTool())
        _global_executor.register_tool(CalculatorTool())
    return _global_executor