# tool_executor.py
"""
Tool execution framework for handling function/tool calls from LLMs.
"""

import asyncio
import hashlib
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    atomic_private_write_bytes,
    open_private_binary,
    secure_private_directory,
)
from tldw_chatbook.Utils.persistent_diagnostics import (
    log_persistent_metadata,
    safe_metadata_token,
)

TOOL_CACHE_FORMAT_VERSION = 1
TOOL_CACHE_MAX_BYTES = 1024 * 1024
_TOOL_CACHE_MAX_DEPTH = 20
_TOOL_CACHE_MAX_NODES = 10_000
_TOOL_CACHE_MAX_STRING_BYTES = 64 * 1024
_TOOL_HISTORY_LIMIT = 100
_persistent_logger = logging.getLogger(__name__)


def _metadata_identity(value: Any) -> str:
    """Return a bounded diagnostic identity without arbitrary payload text."""

    return safe_metadata_token(value)


def _result_metadata(result: Any) -> tuple[str, int]:
    """Return result type and a non-content size approximation."""

    result_type = _metadata_identity(type(result).__name__)
    try:
        result_size = len(result)
    except (TypeError, AttributeError):
        result_size = 0
    return result_type, max(0, int(result_size))


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
                "parameters": self.parameters,
            },
        }


class ToolResultCache:
    """LRU cache for tool execution results with TTL support."""

    def __init__(
        self,
        max_size: int = 100,
        default_ttl: int = 3600,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default time-to-live in seconds
            persist_path: Optional path to persist cache to disk
        """
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.persist_path = Path(persist_path) if persist_path is not None else None
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._load_task = None
        self._legacy_persistence_blocked = False

        # Load cache from disk if persist_path is provided
        if self.persist_path:
            self._load_task = asyncio.create_task(self._load_from_disk())

    def _generate_cache_key(self, tool_name: str, args: dict) -> str:
        """Generate a unique cache key for tool call."""
        # Create a stable string representation of arguments
        args_str = json.dumps(args, sort_keys=True)
        key_str = f"{tool_name}:{args_str}"
        # Use hash for consistent key length
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get(self, tool_name: str, args: dict) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available and not expired.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Cached result or None
        """
        if self._load_task and not self._load_task.done():
            await self._load_task

        cache_key = self._generate_cache_key(tool_name, args)
        async with self._lock:
            if cache_key in self.cache:
                result, expiry_time = self.cache[cache_key]

                # Check if expired
                if time.time() > expiry_time:
                    del self.cache[cache_key]
                    return None

                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                logger.debug(f"Cache hit for tool {tool_name}")
                return result

        return None

    async def set(
        self,
        tool_name: str,
        args: dict,
        result: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """
        Cache a tool execution result.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Execution result
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Wait for initial load if still in progress
        if self._load_task and not self._load_task.done():
            await self._load_task

        cache_key = self._generate_cache_key(tool_name, args)
        ttl = ttl or self.default_ttl
        expiry_time = time.time() + ttl

        async with self._lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and cache_key not in self.cache:
                self.cache.popitem(last=False)

            self.cache[cache_key] = (result, expiry_time)
            self.cache.move_to_end(cache_key)
            logger.debug(f"Cached result for tool {tool_name} (TTL: {ttl}s)")

        if self.persist_path:
            await self._save_to_disk()

    async def clear_expired(self):
        """Remove all expired entries from cache."""
        current_time = time.time()
        async with self._lock:
            expired_keys = [
                key for key, (_, expiry) in self.cache.items() if current_time > expiry
            ]
            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                logger.debug(f"Cleared {len(expired_keys)} expired cache entries")

    async def clear(self):
        """Clear all cached entries."""
        if self._load_task and not self._load_task.done():
            await self._load_task
        async with self._lock:
            self.cache.clear()
            self._legacy_persistence_blocked = False
            logger.info("Tool result cache cleared")

        if self.persist_path:
            await self._save_to_disk()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "persist_path": str(self.persist_path) if self.persist_path else None,
        }

    async def _load_from_disk(self):
        """Load cache from disk if available."""
        if not self.persist_path:
            return

        try:
            secure_private_directory(
                self.persist_path.parent,
                create=True,
                application_owned=True,
            )
            with open_private_binary(self.persist_path) as pinned:
                raw = pinned.stream.read(TOOL_CACHE_MAX_BYTES + 1)
        except FileNotFoundError:
            return
        except PrivatePathError as exc:
            logger.warning(
                "Persistent tool cache load disabled: unsafe target (status={}).",
                exc.result.status.value,
            )
            return
        except OSError as exc:
            logger.warning(
                "Persistent tool cache load failed (error_type={}).",
                type(exc).__name__,
            )
            return

        if len(raw) > TOOL_CACHE_MAX_BYTES:
            logger.warning("Persistent tool cache ignored (reason=oversized).")
            return
        if raw.lstrip()[:1] not in {b"", b"{"}:
            self._legacy_persistence_blocked = True
            logger.warning(
                "Persistent tool cache left inert (reason=legacy_or_non_json)."
            )
            return
        loaded_entries = self._decode_cache_envelope(raw)
        if loaded_entries is None:
            logger.warning("Persistent tool cache ignored (reason=invalid_format).")
            return

        async with self._lock:
            self.cache.update(loaded_entries)
        logger.info(
            "Loaded persistent tool cache entries (entry_count={}).",
            len(loaded_entries),
        )

    async def _save_to_disk(self):
        """Save cache to disk."""
        if not self.persist_path or self._legacy_persistence_blocked:
            return

        try:
            async with self._lock:
                items = list(self.cache.items())
            entries = [
                {
                    "key": key,
                    "expires_at": expiry_time,
                    "result": result,
                }
                for key, (result, expiry_time) in items
                if self._valid_cache_key(key)
                and self._valid_expiry(expiry_time)
                and self._valid_json_value(result)
            ]
            encoded = self._encode_cache_envelope(entries)
            while len(encoded) > TOOL_CACHE_MAX_BYTES and entries:
                entries.pop(0)
                encoded = self._encode_cache_envelope(entries)
            if len(encoded) > TOOL_CACHE_MAX_BYTES:
                logger.warning(
                    "Persistent tool cache save skipped (reason=encoded_size)."
                )
                return
            atomic_private_write_bytes(
                self.persist_path,
                encoded,
                application_owned_directory=self.persist_path.parent,
            )
            logger.debug(
                "Saved persistent tool cache entries (entry_count={}).",
                len(entries),
            )
        except PrivatePathError as exc:
            logger.warning(
                "Persistent tool cache save disabled: unsafe target (status={}).",
                exc.result.status.value,
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Persistent tool cache save failed (error_type={}).",
                type(exc).__name__,
            )

    def _decode_cache_envelope(
        self,
        raw: bytes,
    ) -> list[tuple[str, tuple[Any, float]]] | None:
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(decoded, dict)
            or set(decoded) != {"version", "entries"}
            or decoded["version"] != TOOL_CACHE_FORMAT_VERSION
            or not isinstance(decoded["entries"], list)
            or len(decoded["entries"]) > self.max_size
        ):
            return None

        current_time = time.time()
        seen: set[str] = set()
        loaded: list[tuple[str, tuple[Any, float]]] = []
        for entry in decoded["entries"]:
            if not isinstance(entry, dict) or set(entry) != {
                "key",
                "expires_at",
                "result",
            }:
                return None
            key = entry["key"]
            expiry_time = entry["expires_at"]
            result = entry["result"]
            if (
                not self._valid_cache_key(key)
                or key in seen
                or not self._valid_expiry(expiry_time)
                or not self._valid_json_value(result)
            ):
                return None
            seen.add(key)
            if expiry_time > current_time:
                loaded.append((key, (result, float(expiry_time))))
        return loaded

    @staticmethod
    def _valid_cache_key(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _valid_expiry(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )

    @staticmethod
    def _valid_json_value(value: Any) -> bool:
        remaining_nodes = _TOOL_CACHE_MAX_NODES

        def visit(item: Any, depth: int) -> bool:
            nonlocal remaining_nodes
            remaining_nodes -= 1
            if remaining_nodes < 0 or depth > _TOOL_CACHE_MAX_DEPTH:
                return False
            if item is None or isinstance(item, bool):
                return True
            if isinstance(item, str):
                return len(item.encode("utf-8")) <= _TOOL_CACHE_MAX_STRING_BYTES
            if isinstance(item, int):
                return True
            if isinstance(item, float):
                return math.isfinite(item)
            if isinstance(item, list):
                return all(visit(child, depth + 1) for child in item)
            if isinstance(item, dict):
                return all(
                    isinstance(key, str)
                    and len(key.encode("utf-8")) <= _TOOL_CACHE_MAX_STRING_BYTES
                    and visit(child, depth + 1)
                    for key, child in item.items()
                )
            return False

        return visit(value, 0)

    @staticmethod
    def _encode_cache_envelope(entries: list[dict[str, Any]]) -> bytes:
        return json.dumps(
            {
                "version": TOOL_CACHE_FORMAT_VERSION,
                "entries": entries,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


class ToolExecutor:
    """Manages tool registration and execution with safety measures."""

    def __init__(
        self,
        timeout_seconds: int = 30,
        max_workers: int = 4,
        enable_cache: bool = False,
        cache_size: int = 100,
        cache_ttl: int = 3600,
        cache_persist_path: Optional[Path] = None,
    ):
        """
        Initialize the tool executor.

        Args:
            timeout_seconds: Maximum execution time for each tool
            max_workers: Maximum concurrent tool executions
            enable_cache: Whether to enable result caching
            cache_size: Maximum number of cached results
            cache_ttl: Default cache time-to-live in seconds
            cache_persist_path: Optional path to persist cache to disk
        """
        self.tools: Dict[str, Tool] = {}
        self.timeout_seconds = timeout_seconds
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._execution_history: deque[Dict[str, Any]] = deque(
            maxlen=_TOOL_HISTORY_LIMIT
        )
        self.enable_cache = enable_cache
        self.cache = (
            ToolResultCache(
                max_size=cache_size,
                default_ttl=cache_ttl,
                persist_path=cache_persist_path,
            )
            if enable_cache
            else None
        )

    def register_tool(self, tool: Tool):
        """
        Register a tool for execution.

        Args:
            tool: The tool instance to register
        """
        if not isinstance(tool, Tool):
            raise ValueError(
                f"Tool must be an instance of Tool class, got {type(tool)}"
            )

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
        Execute a single tool call with safety measures and caching.

        Args:
            tool_call: Tool call in OpenAI format

        Returns:
            Dictionary with tool_call_id and either result or error
        """
        tool_call_id = tool_call.get("id", f"call_{datetime.now().timestamp()}")
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name")

        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()

        def record(
            status: str,
            *,
            argument_names: list[str] | None = None,
            unknown_argument_count: int = 0,
            cache_hit: bool = False,
            result: Any = None,
            exception_type: str | None = None,
            error_category: str | None = None,
        ) -> None:
            duration_ms = max(0, int((time.monotonic() - started) * 1000))
            result_type, result_size = (
                _result_metadata(result) if result is not None else ("none", 0)
            )
            execution_record: dict[str, Any] = {
                "tool_name": _metadata_identity(tool_name),
                "status": status,
                "started_at": started_at,
                "duration_ms": duration_ms,
                "argument_names": sorted(argument_names or []),
                "unknown_argument_count": max(0, unknown_argument_count),
                "cache_hit": cache_hit,
                "result_type": result_type,
                "result_size": result_size,
            }
            if exception_type is not None:
                execution_record["exception_type"] = _metadata_identity(
                    exception_type
                )
            if error_category is not None:
                execution_record["error_category"] = _metadata_identity(
                    error_category
                )
            self._execution_history.append(execution_record)
            log_persistent_metadata(
                _persistent_logger,
                logging.INFO if status in {"success", "cached"} else logging.WARNING,
                "tool_execution",
                tool_name=execution_record["tool_name"],
                status=status,
                duration_ms=duration_ms,
                argument_names=execution_record["argument_names"],
                unknown_argument_count=unknown_argument_count,
                cache_hit=cache_hit,
                result_type=result_type,
                result_size=result_size,
                **(
                    {"exception_type": execution_record["exception_type"]}
                    if exception_type is not None
                    else {}
                ),
                **(
                    {"error_category": execution_record["error_category"]}
                    if error_category is not None
                    else {}
                ),
            )

        # Validate tool exists
        if not tool_name or tool_name not in self.tools:
            error_msg = f"Unknown tool: {tool_name}"
            logger.error("Tool execution rejected: unknown tool")
            record("error", error_category="unknown_tool")
            return {"tool_call_id": tool_call_id, "error": error_msg}

        # Parse arguments
        try:
            args_str = function_data.get("arguments", "{}")
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON arguments: {e}"
            logger.error("Tool arguments could not be parsed")
            record(
                "parse_error",
                exception_type=type(e).__name__,
                error_category="invalid_json",
            )
            return {"tool_call_id": tool_call_id, "error": error_msg}

        tool = self.tools[tool_name]
        schema = tool.parameters if isinstance(tool.parameters, dict) else {}
        properties = schema.get("properties", {})
        registered_names = set(properties) if isinstance(properties, dict) else set()
        supplied_names = set(args) if isinstance(args, dict) else set()
        argument_names = sorted(
            _metadata_identity(name)
            for name in supplied_names & registered_names
            if _metadata_identity(name) != "invalid"
        )
        unknown_argument_count = len(supplied_names - registered_names)

        # Check cache if enabled
        if self.enable_cache and self.cache:
            cached_result = await self.cache.get(tool_name, args)
            if cached_result is not None:
                record(
                    "cached",
                    argument_names=argument_names,
                    unknown_argument_count=unknown_argument_count,
                    cache_hit=True,
                    result=cached_result,
                )
                logger.info("Tool result retrieved from cache")
                return {
                    "tool_call_id": tool_call_id,
                    "result": cached_result,
                    "cached": True,
                }

        # Execute with timeout
        try:
            logger.info("Executing registered tool")

            # Run the async tool in the executor with timeout
            result = await asyncio.wait_for(
                tool.execute(**args), timeout=self.timeout_seconds
            )

            # Cache the result if caching is enabled
            if self.enable_cache and self.cache:
                # Determine if this tool's results should be cached
                # Some tools (like datetime) might have short TTLs
                ttl = None  # Use default TTL
                if tool_name in ["get_current_datetime"]:
                    ttl = 60  # Cache datetime for only 1 minute
                elif tool_name in ["calculator"]:
                    ttl = 3600 * 24  # Cache calculations for 24 hours

                await self.cache.set(tool_name, args, result, ttl)

            record(
                "success",
                argument_names=argument_names,
                unknown_argument_count=unknown_argument_count,
                result=result,
            )
            logger.info("Tool completed successfully")
            return {"tool_call_id": tool_call_id, "result": result}

        except asyncio.TimeoutError:
            error_msg = f"Tool execution timed out after {self.timeout_seconds} seconds"
            logger.error("Tool execution timed out")
            record(
                "timeout",
                argument_names=argument_names,
                unknown_argument_count=unknown_argument_count,
                exception_type="TimeoutError",
                error_category="timeout",
            )
            return {"tool_call_id": tool_call_id, "error": error_msg}

        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error("Tool execution failed (exception_type={})", type(e).__name__)
            record(
                "error",
                argument_names=argument_names,
                unknown_argument_count=unknown_argument_count,
                exception_type=type(e).__name__,
                error_category="execution_failed",
            )
            return {"tool_call_id": tool_call_id, "error": error_msg}

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
                processed_results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "error": f"Execution failed: {str(result)}",
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent tool execution history."""
        if limit <= 0:
            return []
        return [record.copy() for record in list(self._execution_history)[-limit:]]

    def clear_history(self):
        """Clear execution history."""
        self._execution_history.clear()

    async def clear_cache(self):
        """Clear the result cache."""
        if self.cache:
            await self.cache.clear()

    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache:
            stats = self.cache.get_stats()
            # Clean expired entries first
            await self.cache.clear_expired()
            return stats
        return None

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


# Create a global tool executor instance
_global_executor = None


def get_tool_executor() -> ToolExecutor:
    """Get the global tool executor instance."""
    global _global_executor
    if _global_executor is None:
        from ..config import get_cli_setting

        # Get tool configuration
        tools_config = get_cli_setting("tools", {})

        # Extract cache settings
        enable_cache = tools_config.get("cache_enabled", False)
        cache_size = tools_config.get("cache_max_size", 100)
        cache_ttl = tools_config.get("cache_default_ttl", 3600)
        timeout = tools_config.get("timeout_seconds", 30)
        max_workers = tools_config.get("max_workers", 4)

        # Determine cache persist path if enabled
        cache_persist_path = None
        if enable_cache and tools_config.get("cache_persist", True):
            from ..config import USER_DATA_DIR

            cache_dir = Path(USER_DATA_DIR) / "tool_cache"
            cache_persist_path = cache_dir / "tool_results.cache"

        # Create executor with settings
        _global_executor = ToolExecutor(
            timeout_seconds=timeout,
            max_workers=max_workers,
            enable_cache=enable_cache,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            cache_persist_path=cache_persist_path,
        )

        # Register built-in tools based on config
        # Default to enabled if not specified
        if tools_config.get("get_current_datetime_enabled", True):
            _global_executor.register_tool(DateTimeTool())

        if tools_config.get("calculator_enabled", True):
            _global_executor.register_tool(CalculatorTool())

        # Register web search tool if enabled
        if tools_config.get(
            "web_search_enabled", False
        ):  # Default to disabled for safety
            try:
                from .web_search_tool import WebSearchTool

                _global_executor.register_tool(WebSearchTool())
                logger.info("WebSearchTool registered successfully")
            except ImportError as e:
                logger.warning(f"Could not import WebSearchTool: {e}")
            except Exception as e:
                logger.error(f"Error registering WebSearchTool: {e}")

        # Register file operation tools if enabled
        if tools_config.get(
            "read_file_enabled", False
        ):  # Default to disabled for safety
            try:
                from .file_operation_tools import ReadFileTool

                _global_executor.register_tool(ReadFileTool())
                logger.info("ReadFileTool registered successfully")
            except ImportError as e:
                logger.warning(f"Could not import ReadFileTool: {e}")
            except Exception as e:
                logger.error(f"Error registering ReadFileTool: {e}")

        if tools_config.get(
            "list_directory_enabled", False
        ):  # Default to disabled for safety
            try:
                from .file_operation_tools import ListDirectoryTool

                _global_executor.register_tool(ListDirectoryTool())
                logger.info("ListDirectoryTool registered successfully")
            except ImportError as e:
                logger.warning(f"Could not import ListDirectoryTool: {e}")
            except Exception as e:
                logger.error(f"Error registering ListDirectoryTool: {e}")

        if tools_config.get(
            "write_file_enabled", False
        ):  # Default to disabled for safety
            try:
                from .file_operation_tools import WriteFileTool

                _global_executor.register_tool(WriteFileTool())
                logger.info("WriteFileTool registered successfully")
            except ImportError as e:
                logger.warning(f"Could not import WriteFileTool: {e}")
            except Exception as e:
                logger.error(f"Error registering WriteFileTool: {e}")

        # Register RAG search tool if enabled
        if tools_config.get(
            "rag_search_enabled", False
        ):  # Default to disabled for safety
            try:
                from .rag_search_tool import RAGSearchTool

                _global_executor.register_tool(RAGSearchTool())
                logger.info("RAGSearchTool registered successfully")
            except ImportError as e:
                logger.warning(f"Could not import RAGSearchTool: {e}")
            except Exception as e:
                logger.error(f"Error registering RAGSearchTool: {e}")

        # Register note management tools if enabled
        if tools_config.get(
            "create_note_enabled", False
        ):  # Default to disabled for safety
            try:
                from .note_management_tools import CreateNoteTool

                _global_executor.register_tool(CreateNoteTool())
                logger.info("CreateNoteTool registered successfully")
            except ImportError as e:
                logger.warning(f"Could not import CreateNoteTool: {e}")
            except Exception as e:
                logger.error(f"Error registering CreateNoteTool: {e}")

        if tools_config.get(
            "search_notes_enabled", False
        ):  # Default to disabled for safety
            try:
                from .note_management_tools import SearchNotesTool

                _global_executor.register_tool(SearchNotesTool())
                logger.info("SearchNotesTool registered successfully")
            except ImportError as e:
                logger.warning(f"Could not import SearchNotesTool: {e}")
            except Exception as e:
                logger.error(f"Error registering SearchNotesTool: {e}")

        if tools_config.get(
            "update_note_enabled", False
        ):  # Default to disabled for safety
            try:
                from .note_management_tools import UpdateNoteTool

                _global_executor.register_tool(UpdateNoteTool())
                logger.info("UpdateNoteTool registered successfully")
            except ImportError as e:
                logger.warning(f"Could not import UpdateNoteTool: {e}")
            except Exception as e:
                logger.error(f"Error registering UpdateNoteTool: {e}")

        # Register code audit tool if enabled
        if tools_config.get(
            "code_audit_enabled", True
        ):  # Default to enabled for monitoring
            try:
                from .code_audit_tool import CodeAuditTool

                _global_executor.register_tool(CodeAuditTool())
                logger.info("CodeAuditTool registered successfully")

                # Try to install automatic file operation hooks
                try:
                    from .file_operation_hooks import install_claude_code_hooks

                    hooks_installed = install_claude_code_hooks()
                    logger.info(f"File operation hooks installed: {hooks_installed}")
                except Exception as hook_error:
                    logger.warning(
                        f"Could not install automatic file operation hooks: {hook_error}"
                    )

            except ImportError as e:
                logger.warning(f"Could not import CodeAuditTool: {e}")
            except Exception as e:
                logger.error(f"Error registering CodeAuditTool: {e}")

        logger.info(
            f"ToolExecutor initialized with: timeout={timeout}s, workers={max_workers}, cache={'enabled' if enable_cache else 'disabled'}"
        )

    return _global_executor


def reload_tool_executor():
    """Reload the tool executor with updated configuration."""
    global _global_executor
    if _global_executor is not None:
        # Shutdown the old executor
        _global_executor.executor.shutdown(wait=False)
        _global_executor = None

    # This will create a new executor with updated config
    return get_tool_executor()
