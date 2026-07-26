# tool_executor.py
"""Dependency-free built-in tools plus the ``Tool`` re-export.

The dispatcher that used to live here (``ToolExecutor``, ``ToolResultCache``,
``get_tool_executor``/``reload_tool_executor``, and the config-driven tool
registration block) served a legacy chat path that no longer exists. It had
no remaining callers, and its own config read (``get_cli_setting("tools",
{})``) mis-slotted ``{}`` into the *key* parameter and so always returned
``{}`` (TASK-547). Agent tools now run exclusively through the
permission-gated runtime (``Tools/base.py`` + ``Agents/tool_catalog.py`` +
the ``agent:builtin`` permission matrix); nothing may dispatch a tool call
outside that path.

What remains: the ``Tool`` re-export for callers not yet repointed to
``.base``, and the two dependency-free built-in tools (``DateTimeTool``,
``CalculatorTool``) that ``Agents/tool_catalog.py`` always registers.
"""

from datetime import datetime
from typing import Any, Dict

from .base import Tool  # re-exported for callers not yet repointed


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
                    "default": "UTC",
                }
            },
            "required": [],
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
                "unix_timestamp": int(now.timestamp()),
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
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')",
                }
            },
            "required": ["expression"],
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
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
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
            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree.body)

            return {
                "expression": expression,
                "result": result,
                "result_type": type(result).__name__,
            }
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")
