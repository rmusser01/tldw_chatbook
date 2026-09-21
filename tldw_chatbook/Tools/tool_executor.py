# tool_executor.py
"""
Tool ABC and the two always-on built-in tools (datetime, calculator).

This module used to also hold a full execution framework -- a cache
class, a class managing tool registration/execution, and their global
accessor functions -- for routing LLM function/tool calls. That
framework -- "System A" -- had zero production callers of its batch
execution entry point; the agent runtime (System B) executes tools
through its own local_runtime_delegate instead. It was retired in
TASK-545 P3. What remains here is the load-bearing half: the Tool ABC
that Agents/builtin_tool_gate.py and Agents/tool_catalog.py build on,
and the two built-in tools they wrap.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any


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

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Risk classes for the permission gate, e.g. ``("mutates",)``.

        Concrete with an empty default so every existing subclass keeps
        working unchanged. For tools reached through the agent runtime
        the vocabulary is the permission store's
        ``BUILTIN_HIGH_RISK_TAGS`` (``mutates``/``process``/``reads``/
        ``network``) -- a tool tagged with one of those has an INHERITED
        ``allow`` floored to ``ask`` by ``resolve_builtin_state``. MCP
        tools are resolved against the narrower ``HIGH_RISK_TAGS``
        instead. Tools with no elevated risk leave this empty.

        Returns:
            A tuple of risk tag strings drawn from
            ``BUILTIN_HIGH_RISK_TAGS``; empty for a tool with no
            elevated risk.
        """
        return ()

    @property
    def timeout_seconds(self) -> float:
        """Per-call wall-clock ceiling, or 0 to use the run's default.

        Concrete with a 0 default so every existing subclass is unchanged.
        A tool whose real work legitimately outlasts
        ``RunBudget.max_tool_call_seconds`` (e.g. a long-running external
        operation) raises this; a tool whose work must be cut short sooner
        than that default lowers it. Note the timeout ABANDONS the worker
        thread rather than killing it, so a tool raising this must be
        idempotent or must say so in its timeout message.

        Returns:
            Seconds, or 0.0 to defer to the run budget.
        """
        return 0.0

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
                "parameters": self.parameters,
            },
        }


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


# TASK-32806.4: bounds for the always-on calculator. Sized to be far above
# any arithmetic a person would ask for and far below anything that costs
# real time or memory: 65,536 bits is a ~19,700-digit integer, and the
# string cap is a screenful of screenfuls. They are checked against the
# OPERANDS, before the result is built.
MAX_RESULT_BITS = 1 << 16
MAX_EXPONENT = 1 << 16
MAX_STRING_RESULT_LENGTH = 10_000
MAX_EXPRESSION_LENGTH = 1_000


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

        def _reject_if_oversized(op, left, right) -> None:
            """Refuse an operand pair whose RESULT would be enormous.

            TASK-32806.4: this tool is always on, inherits allow, and
            evaluates whatever the model sends, so a prompt injection is
            enough to reach it. Measured before this check: ``'ab' * 10**7``
            allocated 20 MB in 1 ms and ``7 ** (10**6)`` built a 0.37 MB
            integer in 0.1 s, with bignum powers scaling superlinearly. The
            enclosing timeout does not help -- it abandons the thread rather
            than stopping it, so the CPU stays pinned for the life of the
            process.

            The size is predicted from the operands instead of being
            discovered by computing the result, which is the whole point:
            by the time the result exists the damage is done.
            """
            if op is ast.Mult:
                for text, count in ((left, right), (right, left)):
                    # bytes/bytearray repetition (b'x' * 10**9) blows up memory
                    # exactly like str repetition (Qodo #2).
                    if isinstance(text, (str, bytes, bytearray)) and isinstance(
                        count, int
                    ):
                        if len(text) * max(count, 0) > MAX_STRING_RESULT_LENGTH:
                            raise ValueError(
                                "sequence repetition would produce more than "
                                f"{MAX_STRING_RESULT_LENGTH} characters"
                            )
                        return
                if isinstance(left, int) and isinstance(right, int):
                    if left.bit_length() + right.bit_length() > MAX_RESULT_BITS:
                        raise ValueError(
                            "multiplication would produce a number larger than "
                            f"{MAX_RESULT_BITS} bits"
                        )
                return
            if op is ast.Mod and isinstance(left, (str, bytes, bytearray)):
                # '%1000000000s' % 'x' allocates an enormous string. In a
                # calculator, %% is numeric modulo, never string formatting
                # (Qodo #2).
                raise ValueError(
                    "string/bytes formatting with %% is not a calculator operation"
                )
            if op is ast.Pow:
                if not isinstance(right, int) or not isinstance(left, int):
                    return  # floats overflow to a normal OverflowError
                if right < 0:
                    return
                if right > MAX_EXPONENT:
                    raise ValueError(
                        f"exponent {right} exceeds the maximum of {MAX_EXPONENT}"
                    )
                if max(left.bit_length(), 1) * right > MAX_RESULT_BITS:
                    raise ValueError(
                        "exponentiation would produce a number larger than "
                        f"{MAX_RESULT_BITS} bits"
                    )

        def safe_eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                op = type(node.op)
                if op not in allowed_operators:
                    raise ValueError(f"Operator {op.__name__} not allowed")
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                _reject_if_oversized(op, left, right)
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
            # Qodo #4: reject a non-string model argument at the boundary
            # before it reaches parsing/business logic.
            if not isinstance(expression, str):
                raise ValueError("expression must be a string")
            if len(expression) > MAX_EXPRESSION_LENGTH:
                raise ValueError(
                    f"expression exceeds {MAX_EXPRESSION_LENGTH} characters"
                )
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
