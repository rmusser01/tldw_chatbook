"""Small adapter from Chatbook's decorators to the strict MCP gateway."""

from __future__ import annotations

import copy
import inspect
import re
from collections.abc import Awaitable, Callable
from typing import Any

from mcp_unified.gateway import (
    GatewayJSONValue,
    GatewayRequestContext,
    GatewayToolExecutionError,
)


_TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_ToolHandler = Callable[..., Awaitable[GatewayJSONValue]]


class ChatbookGatewayRuntime:
    """Register and dispatch Chatbook's standalone built-in tools."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        tool_descriptors: list[dict[str, Any]],
    ) -> None:
        self.name = self._bounded_identity(name)
        self.version = self._bounded_identity(version)
        if not isinstance(tool_descriptors, list):
            raise ValueError("tool_descriptors must be a list")

        self._tool_descriptors: dict[str, dict[str, Any]] = {}
        for descriptor in tool_descriptors:
            descriptor_copy = self._validated_descriptor(descriptor)
            descriptor_name = descriptor_copy["name"]
            if descriptor_name in self._tool_descriptors:
                raise ValueError(f"duplicate tool descriptor: {descriptor_name}")
            self._tool_descriptors[descriptor_name] = descriptor_copy

        self._tool_handlers: dict[str, _ToolHandler] = {}
        self._finalized = False

    @staticmethod
    def _bounded_identity(value: object) -> str:
        if not isinstance(value, str) or not value or len(value) > 512:
            raise ValueError("runtime name and version must be bounded strings")
        return value

    @staticmethod
    def _validated_descriptor(descriptor: object) -> dict[str, Any]:
        if not isinstance(descriptor, dict):
            raise ValueError("tool descriptor must be a dictionary")
        name = descriptor.get("name")
        if not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None:
            raise ValueError("tool descriptor name is invalid")
        description = descriptor.get("description")
        if (
            not isinstance(description, str)
            or not description
            or len(description) > 4_096
        ):
            raise ValueError("tool descriptor description must be a bounded string")
        schema = descriptor.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("tool inputSchema must have type object")
        if schema.get("additionalProperties") is not False:
            raise ValueError("tool inputSchema additionalProperties must be false")
        if not isinstance(schema.get("properties"), dict) or not isinstance(
            schema.get("required"), list
        ):
            raise ValueError("tool inputSchema properties and required are invalid")
        return copy.deepcopy(descriptor)

    def tool(
        self, *, name: str | None = None
    ) -> Callable[[_ToolHandler], _ToolHandler]:
        """Return a decorator that records one async built-in handler."""
        if self._finalized:
            raise RuntimeError("runtime is finalized")

        def decorator(handler: _ToolHandler) -> _ToolHandler:
            if self._finalized:
                raise RuntimeError("runtime is finalized")
            handler_name = handler.__name__ if name is None else name
            if (
                not isinstance(handler_name, str)
                or _TOOL_NAME.fullmatch(handler_name) is None
            ):
                raise ValueError("tool handler name is invalid")
            if not inspect.iscoroutinefunction(handler):
                raise ValueError("tool handler must be async")
            if handler_name in self._tool_handlers:
                raise ValueError(f"duplicate tool handler: {handler_name}")
            self._tool_handlers[handler_name] = handler
            return handler

        return decorator

    def finalize(self) -> None:
        """Validate exact descriptor/handler identity and freeze registration."""
        if set(self._tool_descriptors) != set(self._tool_handlers):
            raise ValueError("tool descriptor and handler names must match exactly")
        self._finalized = True

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("runtime must be finalized before serving")

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return detached descriptors in their constructor order."""
        self._require_finalized()
        return copy.deepcopy(list(self._tool_descriptors.values()))

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> GatewayJSONValue:
        """Call a registered built-in and return its application value raw."""
        self._require_finalized()
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise GatewayToolExecutionError(
                "Tool not found",
                reason_code="tool_not_found",
            )
        return await handler(**arguments)
